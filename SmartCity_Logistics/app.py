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

st.set_page_config(page_title="Smart City Logistics 3D", layout="wide")
st.title("Optimización de Rutas - Entorno Semi-3D")

# Coordenadas de Alta Vista, Ciudad Guayana
LAT, LON = 8.2986, -62.7232
NUM_POINTS = 5  # Número de puntos a generar aleatoriamente

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
        "cost_factor": 3, # Más penalizado por distancia corta
        "color": "[0, 80, 255, 255]", # Azul
        "emoji": "🚛"
    },
    "SupraGuayana (Aumark)": {
        "capacity_kg": 2000, 
        "speed_kmh": 40, 
        "efficiency_kml": 8, 
        "cost_factor": 2, # Costo Medio
        "color": "[255, 0, 255, 255]", # Magenta
        "emoji": "🚚"
    },
    "Eco Bolívar (Trimoto)": {
        "capacity_kg": 400, 
        "speed_kmh": 25, 
        "efficiency_kml": 25, 
        "cost_factor": 1, # El más barato para el algoritmo
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
def fetch_street_network(lat, lon, radius=800):
    return ox.graph_from_point((lat, lon), dist=radius, network_type='drive')

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
def fetch_building_data(lat, lon, radius=800):
    tags = {"building": True}
    gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
    gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    # CORRECCIÓN DEL ERROR DE LA IMAGEN:
    alturas_random = pd.Series(np.random.randint(10, 45, size=len(gdf)), index=gdf.index)
    gdf['height'] = gdf.get('height', pd.Series(dtype='float64')).fillna(alturas_random)
    
    return json.loads(gdf.to_json())

def generate_pdf_report(vehicle_name, vehicle_data, km, mins, liters, route_sequence):
    """Genera un archivo PDF binario en memoria con el reporte logístico"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=16, style='B')
    pdf.cell(200, 10, text="Reporte de Optimizacion Logistica (SmartCity)", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="1. Detalles de la Flota Vehicular", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    pdf.cell(200, 8, text=f"Modelo Seleccionado: {vehicle_name}".encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Velocidad Operimental: {vehicle_data['speed_kmh']} km/h", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Rendimiento de Combustible: {vehicle_data['efficiency_kml']} km/L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="2. Metricas del Viaje", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=11)
    pdf.cell(200, 8, text=f"Distancia Total de Ruta: {km:.2f} km", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Tiempo Estimado de Viaje: {mins:.1f} minutos", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(200, 8, text=f"Consumo de Combustible Proyectado: {liters:.2f} Litros", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    
    pdf.set_font("helvetica", size=12, style='B')
    pdf.cell(200, 10, text="3. Secuencia de Puntos de Entrega (Coordenadas)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", size=10)
    
    for idx, (step_name, lat, lon) in enumerate(route_sequence):
        line = f"Paso {idx}: {step_name} -> Lat: {lat:.5f}, Lon: {lon:.5f}"
        pdf.cell(200, 6, text=line.encode('latin-1', 'replace').decode('latin-1'), new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    pdf.set_font("helvetica", size=9, style='I')
    pdf.cell(200, 10, text="Generado por SmartCity Logistics TSP Auto-Solver", new_x="LMARGIN", new_y="NEXT", align='C')
    
    # Retornar como byte string
    return bytes(pdf.output())

with st.spinner("Generando ciudad 3D y red vial..."):
    geojson_data = fetch_building_data(LAT, LON)
    G = fetch_street_network(LAT, LON)

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
    num_pts = st.sidebar.slider("Nº de Puntos:", min_value=3, max_value=15, value=NUM_POINTS)
    if st.sidebar.button("Generar Coordenadas"):
        nodes, coords = generate_and_snap_points(G, LAT, LON, num_pts)
        st.session_state['nodes'] = nodes
        st.session_state['coords'] = coords
        st.session_state['solved'] = False

elif input_mode == "Personalizadas":
    st.sidebar.write("Edita la tabla. Máximo 10 paradas.")
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
        if len(valid_df) > 10:
            st.sidebar.error("Máximo 10 coordenadas permitidas para evitar sobrecarga del solver de demo.")
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
layers_to_render = [layer_buildings]
chart_placeholder = st.empty()


if 'nodes' in st.session_state and 'coords' in st.session_state:
    nodes = st.session_state['nodes']
    coords = st.session_state['coords']
    
    df_points = pd.DataFrame({
        "lon": coords[:, 1],
        "lat": coords[:, 0],
        "index": ["Depot (Origen)"] + [f"Parada {i}" for i in range(1, len(coords))]
    })
    
    layer_points = pdk.Layer(
        "ColumnLayer",
        df_points,
        get_position="[lon, lat]",
        get_elevation=40,    # Torre de 40m
        elevation_scale=1,
        radius=15,           # Ancho del poste
        get_fill_color=POINT_COLOR,
        pickable=True,
        auto_highlight=True,
    )
    layers_to_render.append(layer_points)

    st.sidebar.markdown("---")
    if st.sidebar.button("Calcular Rutas de Flota", type="primary"):
        with st.spinner("Despachando unidades y calculando rutas VRP..."):
            dist_matrix = network_distance_matrix(G, nodes)
            
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
            
            if len(active_capacities) == 0:
                st.sidebar.error("⚠️ Debes habilitar al menos un vehículo en la flota.")
            else:
                optimal_routes = solve_vrp(dist_matrix, demands, active_capacities, active_costs, depot_index=0)
                
                # Validar si hubo solución real (alguna ruta no vacía)
                has_solution = any(len(r) > 0 for r in optimal_routes)
                
                if has_solution:
                    st.session_state['optimal_routes'] = optimal_routes
                    st.session_state['active_vehicles'] = active_vehicles_info
                    st.session_state['demands'] = demands
                    st.session_state['solved'] = True
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
        
        route_sequence_data = [] # Para el PDF
        csv_data = []            # Para el CSV
        
        layers_vrp_paths = []    # Para PyDeck
        
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
            
            # Simulaciones dadas por el vehículo asignado
            # Gasto = Distancia / Eficiencia * (Tráfico Random)
            traffic_factor = np.random.uniform(1.0, 1.3)
            liters = (km / v_data["efficiency_kml"]) * traffic_factor
            cost_usd = liters * 0.5 # $0.5 USD por litro
            
            total_fleet_liters += liters
            total_fleet_cost_usd += cost_usd

        
            with st.sidebar.expander(f"{v_data['emoji']} Vehículo {v_idx+1}: {v_name}", expanded=False):
                st.write(f"**Recorrido:** {km:.2f} km")
                st.write(f"**Combustible:** {liters:.1f} L (${cost_usd:.2f} USD)")
                
                # Carga total recogida
                total_load = sum(demands[node] for node in route if node != 0)
                st.write(f"**Carga Transportada:** {total_load} kg / {v_data['capacity_kg']} kg")
                
                st.write(f"**Orden:** {' -> '.join(map(str, route))}")
                
                # Datos para exportar
                for step, node_idx in enumerate(route):
                    lat, lon = coords[node_idx][0], coords[node_idx][1]
                    name_point = f"Parada {node_idx} ({demands[node_idx]} kg)" if node_idx != 0 else "Base/Vertedero"
                    route_sequence_data.append((f"[V{v_idx+1}] " + name_point, lat, lon))
                    csv_data.append({"Vehiculo": v_name, "Paso": step, "Nombre": name_point, "Latitud": lat, "Longitud": lon, "Carga_Kg": demands[node_idx]})

            # Geometría Fija para este vehículo
            df_path = construct_full_geometry(G, nodes, route)
            if not df_path.empty:
                layer_path = pdk.Layer(
                    "PathLayer",
                    df_path,
                    width_scale=2,
                    width_min_pixels=4,
                    width_max_pixels=8,
                    get_color=v_data["color"],
                    get_path="path",
                    pickable=True,
                    line_joint_rounded=True,
                    line_cap_rounded=True
                )
                layers_vrp_paths.append(layer_path)

        st.sidebar.info(f"**Gasto Total Flota (Gasolina):** ${total_fleet_cost_usd:.2f} USD")

        # --- SECCIÓN DE EXPORTACIÓN ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Exportar Reporte de Flota")
        
        df_csv = pd.DataFrame(csv_data)
        csv_bytes = df_csv.to_csv(index=False).encode('utf-8')
        
        pdf_bytes = generate_pdf_report("Flota Heterogénea", {"speed_kmh": "Variable", "efficiency_kml": "Variable"}, 0, 0, total_fleet_liters, route_sequence_data)
        
        col_dl1, col_dl2 = st.sidebar.columns(2)
        with col_dl1:
            st.download_button(label="📄 Reporte PDF", data=pdf_bytes, file_name="reporte_flota_vrp.pdf", mime="application/pdf")
        with col_dl2:
            st.download_button(label="📊 Rutas CSV", data=csv_bytes, file_name="rutas_flota.csv", mime="text/csv")
        st.sidebar.markdown("---")

        layers_to_render.extend(layers_vrp_paths)
        
        # ANIMACIÓN (Pendiente de adaptación multiplata, se deja un placeholder)
        if st.sidebar.button("🚗 Ver Animación (V1)"):
            st.sidebar.warning("La animación multi-vehículo simultánea requiere WebGL concurrente avanzado. Se mostrarán solo las pistas estáticas por ahora.")


# Render estático predeterminado
chart_placeholder.pydeck_chart(pdk.Deck(
    layers=layers_to_render, 
    initial_view_state=view_state, 
    map_style=MAP_STYLE,
    tooltip={"html": "<b>{index}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
))
