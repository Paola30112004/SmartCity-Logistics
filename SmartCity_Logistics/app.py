import streamlit as st
import pydeck as pdk
import osmnx as ox
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from fpdf import FPDF
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'venv'))

# Importar el nuevo VRP Solver
from tsp_solver import solve_vrp

# --- OPTIMIZACIÓN: Configuración de Caché Persistente ---
ox.settings.use_cache = True
ox.settings.cache_folder = "./osmnx_cache"

st.set_page_config(page_title="Smart City Logistics 3D", layout="wide")
st.title("Optimización de Rutas - Entorno Semi-3D")

# Coordenadas Centrales
LAT, LON = 8.2986, -62.7232
NUM_POINTS = 5
MAP_RADIUS = 1500  # Radio del mapa en metros

# Configuración de apariencia
MAP_STYLE = "mapbox://styles/mapbox/navigation-night-v1"
BUILDING_COLOR = "[70, 180, 255, 180]"  # Azul celeste holográfico brillante
PATH_COLOR = "[255, 60, 0, 255]"      # Naranja/Rojo Neón ("Cable Caliente")
POINT_COLOR = "[0, 255, 128, 255]"    # Verde Esmeralda/Cyan brillante

# Modelos de Vehículos de Recolección (Flota Heterogénea VRP)
FLEET_TYPES = {
    "Fospuca (Compactador)": {
        "capacity_kg": 18000, 
        "speed_kmh": 30, 
        "efficiency_kml": 3, 
        "cost_factor": 1.0, # Prioridad Base
        "color": "[0, 80, 255, 255]", # Azul
        "emoji": "🚛"
    },
    "SupraGuayana (Aumark)": {
        "capacity_kg": 2000, 
        "speed_kmh": 40, 
        "efficiency_kml": 8, 
        "cost_factor": 1.5, # Ligeramente más caro por km
        "color": "[255, 0, 255, 255]", # Magenta
        "emoji": "🚚"
    },
    "Eco Bolívar (Trimoto)": {
        "capacity_kg": 400, 
        "speed_kmh": 25, 
        "efficiency_kml": 25, 
        "cost_factor": 2.0, # Más caro por km (para forzar su uso solo en paradas necesarias)
        "color": "[0, 255, 128, 255]", # Verde
        "emoji": "🛵"
    }
}

def haversine_dist_meters(lon1, lat1, lon2, lat2):
    """Calcula distancia entre 2 puntos en metros para el simulador visual"""
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

@st.cache_data
def fetch_street_network(lat, lon, radius=MAP_RADIUS):
    return ox.graph_from_point((lat, lon), dist=radius, network_type='drive')

@st.cache_data
def get_address_from_graph(_G, lat, lon):
    """Obtiene el nombre de la calle más cercana directamente del grafo OSM"""
    try:
        # Encontrar la arista (calle) más cercana
        nearest_edge = ox.distance.nearest_edges(_G, X=lon, Y=lat)
        # nearest_edge es (u, v, key)
        edge_data = _G.get_edge_data(nearest_edge[0], nearest_edge[1], nearest_edge[2])
        street_name = edge_data.get('name', 'Calle sin nombre')
        if isinstance(street_name, list):
            street_name = " / ".join(str(s) for s in street_name)
        return str(street_name)
    except:
        return f"Vía de Alta Vista (Lat: {lat:.4f})"

@st.cache_data
def get_street_geometry(_G):
    """Convierte la red vial en líneas dibujables para el mapa"""
    gdf_edges = ox.graph_to_gdfs(_G, nodes=False, edges=True)
    paths = []
    for _, row in gdf_edges.iterrows():
        if hasattr(row.geometry, 'coords'):
            paths.append({"path": list(row.geometry.coords)})
    return pd.DataFrame(paths)

def generate_and_snap_points(G, center_lat, center_lon, num_points=5, radius_deg=0.01):
    """Genera coordenadas aleatorias y las acopla al nodo de red más cercano"""
    lats = center_lat + np.random.uniform(-radius_deg, radius_deg, num_points)
    lons = center_lon + np.random.uniform(-radius_deg, radius_deg, num_points)
    
    # Encontrar nodos más cercanos en el grafo de OSMnx
    nodes = ox.distance.nearest_nodes(G, X=lons, Y=lats)
    
    # Extraer las coordenadas exactas de esos nodos intersectados
    snapped_coords = []
    for node in nodes:
        snapped_coords.append((G.nodes[node]['y'], G.nodes[node]['x']))
    
    return nodes, np.array(snapped_coords)

# (Las funciones solve_tsp y demás dependencias de or-tools internas en app.py fueron movidas o ya no se usan gracias a solve_vrp)
def network_distance_matrix(G, nodes):
    """
    Calcula la matriz de distancias usando la red vial real.
    Considera si el grafo es fuertemente conexo. Para rutas no alcanzables usa infinito.
    """
    n = len(nodes)
    dist_matrix = np.full((n, n), 9999999) # Usar un costo alto para arcos inalcanzables
    
    for i in range(n):
        # Calcular los caminos más cortos desde el nodo 'i' a todos los demás posibles
        try:
            lengths = nx.single_source_dijkstra_path_length(G, nodes[i], weight='length')
            for j in range(n):
                if nodes[j] in lengths:
                    
                    # SIMULANDO CALLES ESTRECHAS: Agregar penalización aleatoria al tráfico real
                    dist_real = int(lengths[nodes[j]])
                    if np.random.rand() > 0.8: # 20% de probabilidad de ser una calle muy estrecha
                        dist_real += 5000 # Penalizar fuertemente aristas "residenciales"
                        
                    dist_matrix[i, j] = dist_real
        except Exception:
            pass # Si el nodo está totalmente desconectado, queda en 9999999
            
    return dist_matrix

def construct_full_geometry(G, nodes, optimal_route_indices):
    """
    Reconstruye pieza a pieza las calles que conectan los nodos de la ruta óptima.
    """
    full_path_coords = []
    
    if not optimal_route_indices or len(optimal_route_indices) < 2:
        return pd.DataFrame()
        
    for i in range(len(optimal_route_indices) - 1):
        u = nodes[optimal_route_indices[i]]
        v = nodes[optimal_route_indices[i+1]]
        
        try:
            # Encontrar el camino más corto en la red
            path_nodes = nx.shortest_path(G, u, v, weight='length')
            
            # Extraer la geometría (lons, lats para pydeck) de ese segmento
            # En la unión de dos segmentos, removemos el último vértice para no duplicar si agregamos el sgte
            for node in path_nodes[:-1]:
                full_path_coords.append([G.nodes[node]['x'], G.nodes[node]['y']])
                
        except nx.NetworkXNoPath:
            # Si un arco no existe, metemos una linea recta (fallback visual)
            full_path_coords.append([G.nodes[u]['x'], G.nodes[u]['y']])
    
    # Agregar el último nodo del recorrido completo
    last_node = nodes[optimal_route_indices[-1]]
    full_path_coords.append([G.nodes[last_node]['x'], G.nodes[last_node]['y']])
    
    return pd.DataFrame({"path": [full_path_coords]})

@st.cache_data
def fetch_building_data(lat, lon, radius=MAP_RADIUS):
    tags = {"building": True}
    gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
    
    # Filtrar solo polígonos para evitar errores en la simplificación
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    
    # OPTIMIZACIÓN: Simplificar geometrías para que el mapa pese menos y vuele
    # El valor 0.00005 es un buen balance entre detalle y peso
    gdf['geometry'] = gdf.geometry.simplify(tolerance=0.00005, preserve_topology=True)
    
    # CORRECCIÓN DEL ERROR DE LA IMAGEN:
    alturas_random = pd.Series(np.random.randint(10, 45, size=len(gdf)), index=gdf.index)
    gdf['height'] = gdf.get('height', pd.Series(dtype='float64')).fillna(alturas_random)
    
    return json.loads(gdf.to_json())

def generate_pdf_report(vehicle_name, vehicle_data, km_total, mins_total, liters_total, route_sequence, cost_total=0):
    """Genera un archivo PDF binario en memoria con el reporte logístico"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=16, style='B')
    pdf.cell(200, 10, text="Reporte de Optimizacion Logistica (SmartCity)", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="1. Resumen de Flota Vehicular Desplegada", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    
    # Adaptar para reporte VRP o TSP
    if vehicle_name != "Flota Heterogénea":
        pdf.cell(200, 8, text=f"Modelo Seleccionado: {vehicle_name}".encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(200, 8, text="Operacion: Multi-Vehiculo (CVRP)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="2. Metricas Globales del Viaje", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    pdf.cell(200, 8, text=f"Distancia Total de Ruta(s): {km_total:.2f} km", new_x="LMARGIN", new_y="NEXT")
    if mins_total > 0:
        pdf.cell(200, 8, text=f"Tiempo Estimado de Viaje (Mas largo): {mins_total:.1f} minutos", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Consumo de Combustible Proyectado: {liters_total:.2f} Litros", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Costo Total de Operacion: ${cost_total:.2f} USD", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="3. Rutas y Puntos de Entrega (Coordenadas)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    
    for idx, (step_name, lat, lon) in enumerate(route_sequence):
        line = f"{step_name} -> Lat: {lat:.5f}, Lon: {lon:.5f}"
        pdf.cell(200, 6, text=line.encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    pdf.set_font("helvetica", size=9, style='I')
    pdf.cell(200, 10, text="Generado por SmartCity Logistics TSP/VRP Auto-Solver", new_x="LMARGIN", new_y="NEXT", align='C')
    
    # Retornar como byte string
    return bytes(pdf.output())

with st.spinner("Generando ciudad y red vial interactiva..."):
    geojson_data = fetch_building_data(LAT, LON)
    G = fetch_street_network(LAT, LON)
    df_streets = get_street_geometry(G)

layer_streets = pdk.Layer(
    "PathLayer",
    df_streets,
    width_min_pixels=2,
    get_color=[255, 255, 255, 60], # Blanco más sutil para la noche
    get_path="path",
    pickable=False
)

layer_buildings = pdk.Layer(
    "GeoJsonLayer",
    geojson_data,
    opacity=0.8,
    extruded=True,
    wireframe=True,
    get_elevation="properties.height * 2.5",
    get_fill_color=BUILDING_COLOR,
    get_line_color=[100, 100, 200, 30],
    pickable=True
)

st.sidebar.header("Planificador de Rutas (VRP)")

# Selección de Flota Vehicular
st.sidebar.markdown("### Unidades Disponibles:")
fleet_availability = {}
for vehicle_name, data in FLEET_TYPES.items():
    default_val = 1 if "Fospuca" in vehicle_name or "Trimoto" in vehicle_name else 0
    qty = st.sidebar.number_input(f"{data['emoji']} {vehicle_name}", min_value=0, max_value=5, value=default_val)
    fleet_availability[vehicle_name] = qty

# UI de Configuración
st.sidebar.markdown("---")
input_mode = st.sidebar.radio("Modo de Coordenadas:", ["Aleatorias", "Personalizadas"])

coords = None
nodes = None

if input_mode == "Aleatorias":
    num_pts = st.sidebar.slider("Nº de Puntos:", min_value=3, max_value=25, value=NUM_POINTS)
    if st.sidebar.button("Generar Coordenadas"):
        nodes, coords = generate_and_snap_points(G, LAT, LON, num_pts)
        st.session_state['nodes'] = nodes
        st.session_state['coords'] = coords
        st.session_state['solved'] = False

elif input_mode == "Personalizadas":
    st.sidebar.write("Edita la tabla. Máximo 20 paradas.")
    # Datos base para el editor si no existen en estado
    if 'custom_df' not in st.session_state:
        # Se añaden 3 puntos cercanos iniciales de ejemplo
        st.session_state['custom_df'] = pd.DataFrame({
            "Latitud": [LAT, LAT + 0.005, LAT - 0.003],
            "Longitud": [LON, LON + 0.002, LON - 0.004]
        })
    
    edited_df = st.sidebar.data_editor(st.session_state['custom_df'], num_rows="dynamic")
    st.session_state['custom_df'] = edited_df

    if st.sidebar.button("Fijar Coordenadas"):
        valid_df = edited_df.dropna()
        if len(valid_df) > 20:
            st.sidebar.error("Máximo 20 coordenadas permitidas para evitar sobrecarga del solver de demo.")
        elif len(valid_df) < 2:
            st.sidebar.warning("Introduce al menos 2 coordenadas.")
        else:
            raw_lats = valid_df['Latitud'].values
            raw_lons = valid_df['Longitud'].values
            # Hacemos snap a la red real de inmediato
            snapped_nodes = ox.distance.nearest_nodes(G, X=raw_lons, Y=raw_lats)
            snapped_coords = []
            for n in snapped_nodes:
                snapped_coords.append([G.nodes[n]['y'], G.nodes[n]['x']])
            
            st.session_state['nodes'] = snapped_nodes
            st.session_state['coords'] = np.array(snapped_coords)
            st.session_state['solved'] = False
            st.sidebar.success(f"{len(snapped_nodes)} paradas acopladas a esquinas.")

# Estado inicial Renderizado de Vistas
view_state = pdk.ViewState(
    latitude=LAT, 
    longitude=LON, 
    zoom=14.5, 
    pitch=65, 
    bearing=30 
)
layers_to_render = [layer_streets, layer_buildings]
chart_placeholder = st.empty()


if 'nodes' in st.session_state and 'coords' in st.session_state:
    nodes = st.session_state['nodes']
    coords = st.session_state['coords']
    
    # Obtener nombres de calles reales usando el grafo G
    addresses = [get_address_from_graph(G, c[0], c[1]) for c in coords]
    
    df_points = pd.DataFrame({
        "lon": coords[:, 1],
        "lat": coords[:, 0],
        "direccion": addresses,
        "index": ["Base Logística (Vertedero)"] + [f"Parada {i}" for i in range(1, len(coords))]
    })
    
    layer_points = pdk.Layer(
        "ColumnLayer",
        df_points,
        get_position="[lon, lat]",
        get_elevation=60,    # Torre más visible
        elevation_scale=1,
        radius=20,           # Un poco más robusto
        get_fill_color=POINT_COLOR,
        pickable=True,
        auto_highlight=True,
    )
    layers_to_render.append(layer_points)

    st.sidebar.markdown("---")
    if st.sidebar.button("Calcular Rutas de Flota", type="primary"):
        # OPTIMIZACIÓN: Solo recalcular la matriz si los puntos han cambiado
        current_nodes_hash = hash(tuple(nodes))
        prev_nodes_hash = st.session_state.get('last_nodes_hash')
        
        # Limpiar resultados anteriores para evitar confusión
        if 'optimal_routes' in st.session_state: del st.session_state['optimal_routes']
        if 'active_vehicles' in st.session_state: del st.session_state['active_vehicles']
        st.session_state['solved'] = False
        
        with st.spinner("Despachando unidades y calculando rutas VRP..."):
            if current_nodes_hash == prev_nodes_hash and 'dist_matrix' in st.session_state:
                dist_matrix = st.session_state['dist_matrix']
            else:
                dist_matrix = network_distance_matrix(G, nodes)
                st.session_state['dist_matrix'] = dist_matrix
                st.session_state['last_nodes_hash'] = current_nodes_hash
            
            # Preparar demandas aleatorias (simulando basura en kg) para las paradas. Depot = 0.
            demands = [0] + [int(np.random.randint(50, 250)) for _ in range(len(nodes) - 1)]
            
            # Compilar la flota disponible real según el usuario
            active_capacities = []
            active_costs = []
            active_vehicles_info = []
            
            for v_name, qty in fleet_availability.items():
                for _ in range(qty):
                    active_capacities.append(FLEET_TYPES[v_name]["capacity_kg"])
                    active_costs.append(FLEET_TYPES[v_name]["cost_factor"])
                    active_vehicles_info.append((v_name, FLEET_TYPES[v_name]))
            
            num_available_pts = len(nodes) - 1
            if len(active_capacities) == 0:
                st.sidebar.error("⚠️ Debes habilitar al menos un vehículo en la flota.")
            elif len(active_capacities) > num_available_pts:
                st.sidebar.error(f"⚠️ Tienes {len(active_capacities)} vehículos pero solo {num_available_pts} paradas. Reduce la flota o aumenta los puntos.")
            else:
                optimal_routes = solve_vrp(dist_matrix, demands, active_capacities, active_costs, depot_index=0)
                
                # Validar si hubo solución real (alguna ruta no vacía)
                has_solution = any(len(r) > 0 for r in optimal_routes)
                
                if has_solution:
                    st.session_state['optimal_routes'] = optimal_routes
                    st.session_state['active_vehicles'] = active_vehicles_info
                    st.session_state['demands'] = demands
                    st.session_state['solved'] = True
                    st.sidebar.success(f"✓ {sum(1 for r in optimal_routes if len(r)>2)} vehículos en ruta.")
                else:
                    st.sidebar.error("Inviable: Capacidad insuficiente en la flota o puntos desconectados.")

    if st.session_state.get('solved', False):
        optimal_routes = st.session_state['optimal_routes']
        active_vehicles = st.session_state['active_vehicles']
        demands = st.session_state['demands']
        
        # MÉTRICAS Y SIMULACIÓN FÍSICA
        st.sidebar.subheader("Desempeño de la Flota:")
        
        total_fleet_cost_usd = 0
        total_fleet_liters = 0
        total_fleet_km = 0
        fleet_total_time_mins = 0
        max_time_mins = 0
        
        route_sequence_data = [] # Para el PDF
        csv_data = []            # Para el CSV
        
        layers_vrp_paths = []    # Para PyDeck
        vehicle_animation_data = [] # Para animacion simultanea
        
        # Check para nodos no visitados
        visited_nodes = set([0])
        
        for v_idx, route in enumerate(optimal_routes):
            if not route: continue # El camión no se usó
            
            v_name, v_data = active_vehicles[v_idx]
            
            # Calcular distancia real de esta ruta
            dist_m = 0
            for i in range(len(route) - 1):
                dist_m += haversine_dist_meters(
                    coords[route[i]][1], coords[route[i]][0],
                    coords[route[i+1]][1], coords[route[i+1]][0]
                )
            
            km = dist_m / 1000.0
            total_fleet_km += km
            
            # Simulaciones dadas por el vehículo asignado
            traffic_factor = np.random.uniform(1.0, 1.2)
            liters = (km / v_data["efficiency_kml"]) * traffic_factor
            cost_usd = liters * 0.5 # $0.5 USD por litro
            
            # Tiempo del viaje = Distancia / Velocidad
            mins = (km / v_data["speed_kmh"]) * 60.0
            fleet_total_time_mins += mins
            if mins > max_time_mins: max_time_mins = mins
            
            total_fleet_liters += liters
            total_fleet_cost_usd += cost_usd
        
            with st.sidebar.expander(f"{v_data['emoji']} Vehículo {v_idx+1}: {v_name}", expanded=False):
                st.write(f"**Recorrido:** {km:.2f} km")
                st.write(f"**Tiempo Est.:** {mins:.1f} min")
                st.write(f"**Combustible:** {liters:.1f} L (${cost_usd:.2f} USD)")
                
                # Carga total recogida
                total_load = sum(demands[node] for node in route if node != 0)
                st.write(f"**Carga Transportada:** {total_load} kg / {v_data['capacity_kg']} kg")
                
                st.write(f"**Orden:** {' -> '.join(map(str, route))}")
                
                # Datos para exportar
                for step, node_idx in enumerate(route):
                    visited_nodes.add(node_idx)
                    lat, lon = coords[node_idx][0], coords[node_idx][1]
                    name_point = f"Parada {node_idx} ({demands[node_idx]} kg)" if node_idx != 0 else "Base/Vertedero"
                    route_sequence_data.append((f"[Vehículo {v_idx+1}] Paso {step}: {name_point}", lat, lon))
                    csv_data.append({"Vehiculo": v_name, "Paso": step, "Nombre": name_point, "Latitud": lat, "Longitud": lon, "Carga_Kg": demands[node_idx]})

            # Geometría Fija para este vehículo
            df_path = construct_full_geometry(G, nodes, route)
            if not df_path.empty:
                # Variar ligeramente el color si hay muchos del mismo tipo para distinguirlos
                base_color = v_data["color"].replace("[", "").replace("]", "").split(",")
                r, g, b, a = [int(c) for c in base_color]
                # Aplicar un pequeño offset aleatorio al color para evitar superposición perfecta visual
                r = min(255, max(0, r + (v_idx * 20) % 50))
                g = min(255, max(0, g + (v_idx * 30) % 50))
                disting_color = f"[{r}, {g}, {b}, {a}]"
                
                layer_path = pdk.Layer(
                    "PathLayer",
                    df_path,
                    width_scale=2,
                    width_min_pixels=4,
                    width_max_pixels=8,
                    get_color=disting_color,
                    get_path="path",
                    pickable=True,
                    line_joint_rounded=True,
                    line_cap_rounded=True
                )
                layers_vrp_paths.append(layer_path)
                
                # Setup para animacion
                vehicle_animation_data.append({
                    "path_coords": df_path["path"].iloc[0],
                    "emoji": v_data["emoji"],
                    "color": v_data["color"],
                    "speed": (v_data["speed_kmh"] * 1000) / 3600.0, # m/s
                    "current_idx": 0
                })

        st.sidebar.info(f"**Costo Total Gasolina:** ${total_fleet_cost_usd:.2f} USD")
        st.sidebar.info(f"**Tiempo Operativo Total:** {fleet_total_time_mins:.1f} min")
        
        unassigned = set(range(len(nodes))) - visited_nodes
        if unassigned:
            st.sidebar.error(f"⚠️ Alerta: No hubo capacidad para: {list(unassigned)}")

        # --- SECCIÓN DE EXPORTACIÓN ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Exportar Reporte de Flota")
        
        df_csv = pd.DataFrame(csv_data)
        csv_bytes = df_csv.to_csv(index=False).encode('utf-8')
        
        pdf_bytes = generate_pdf_report("Flota Heterogénea", None, total_fleet_km, fleet_total_time_mins, total_fleet_liters, route_sequence_data, total_fleet_cost_usd)
        
        col_dl1, col_dl2 = st.sidebar.columns(2)
        with col_dl1:
            st.download_button(label="📄 Reporte PDF", data=pdf_bytes, file_name="reporte_flota_vrp.pdf", mime="application/pdf")
        with col_dl2:
            st.download_button(label="📊 Rutas CSV", data=csv_bytes, file_name="rutas_flota.csv", mime="text/csv")
        st.sidebar.markdown("---")

        layers_to_render.extend(layers_vrp_paths)
        
        # ANIMACIÓN MULTI-VEHÍCULO
        if st.sidebar.button("🚗 Empezar Animación de Flota"):
            SIMULATION_SPEED_FACTOR = 0.05 
            
            animating = True
            while animating:
                animating = False
                current_layers = [layer_streets, layer_buildings, layer_points]
                
                for v_data in vehicle_animation_data:
                    idx = v_data["current_idx"]
                    path_coords = v_data["path_coords"]
                    
                    if idx < len(path_coords):
                        animating = True
                        pt = path_coords[idx]
                        
                        # Trazo recorrido hasta ahora
                        anim_path = path_coords[:idx+1]
                        anim_df = pd.DataFrame({"path": [anim_path]})
                        anim_layer = pdk.Layer(
                            "PathLayer",
                            anim_df,
                            width_scale=2,
                            width_min_pixels=6,
                            width_max_pixels=12,
                            get_color=v_data["color"],
                            get_path="path",
                            line_joint_rounded=True,
                            line_cap_rounded=True
                        )
                        current_layers.append(anim_layer)
                        
                        # Emoticono de Vehículo
                        car_df = pd.DataFrame({"lon": [pt[0]], "lat": [pt[1]], "icon": [v_data["emoji"]]})
                        car_layer = pdk.Layer(
                            "TextLayer",
                            car_df,
                            get_position="[lon, lat]",
                            get_text="icon",
                            get_size=40,
                            get_color="[255, 255, 255, 255]",
                            get_alignment_baseline="'center'",
                        )
                        current_layers.append(car_layer)
                        
                        # Adelantar frame (simplificado por frame, ignorando fisicas ultra precisas de colision por ahora)
                        v_data["current_idx"] += 2
                        
                if animating:
                    chart_placeholder.pydeck_chart(pdk.Deck(
                        layers=current_layers,
                        initial_view_state=view_state,
                        map_style=MAP_STYLE,
                        views=[pdk.View(type="MapView", controller=True)]
                    ))
                    time.sleep(SIMULATION_SPEED_FACTOR)
            
            st.sidebar.success("✅ Flota llegó a su destino.")


# Render estático predeterminado
chart_placeholder.pydeck_chart(pdk.Deck(
    layers=layers_to_render, 
    initial_view_state=view_state, 
    map_style=MAP_STYLE,
    tooltip={"html": "<b>{index}</b><br/>{direccion}", "style": {"backgroundColor": "steelblue", "color": "white"}}
))
