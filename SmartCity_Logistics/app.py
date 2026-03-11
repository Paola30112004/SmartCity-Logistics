import streamlit as st
import pydeck as pdk
import osmnx as ox
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

st.set_page_config(page_title="Smart City Logistics 3D", layout="wide")
st.title("Optimización de Rutas - Entorno Semi-3D")

# Coordenadas de Alta Vista, Ciudad Guayana
LAT, LON = 8.2986, -62.7232
NUM_POINTS = 5  # Número de puntos a generar aleatoriamente

# Configuración de apariencia
MAP_STYLE = "mapbox://styles/mapbox/navigation-night-v1"
BUILDING_COLOR = "[30, 255, 200, 80]" # Cian neón translúcido
PATH_COLOR = "[255, 140, 0, 255]"     # Naranja brillante continuo
POINT_COLOR = "[255, 20, 147, 255]"   # Rosa neón vibrante

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
                    dist_matrix[i, j] = int(lengths[nodes[j]])
        except Exception:
            pass # Si el nodo está totalmente desconectado, queda en 9999999
            
    return dist_matrix

def solve_tsp(dist_matrix: np.ndarray, depot_index: int = 0) -> list:
    num_locations = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(num_locations, 1, depot_index)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Agregar dimensión para llevar control del límite del vehículo o imprimir la métrica (opcional, internamente podemos sumar la matriz)
    # Por ahora simplemente pedimos encontrar la heurística más corta.
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return [], 0
        
    index = routing.Start(0)
    route = []
    total_distance = 0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        
    route.append(manager.IndexToNode(index))
    
    return route, total_distance

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

with st.spinner("Generando ciudad 3D y red vial..."):
    geojson_data = fetch_building_data(LAT, LON)
    G = fetch_street_network(LAT, LON)

layer_buildings = pdk.Layer(
    "GeoJsonLayer",
    geojson_data,
    opacity=0.4, # Más transparente para que no ahogue las rutas
    extruded=True,
    get_elevation="properties.height * 1.5",
    get_fill_color=BUILDING_COLOR,
    pickable=True
)

st.sidebar.header("Planificador Logístico - TSP")
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
    pitch=55, 
    bearing=15 
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
        "ScatterplotLayer",
        df_points,
        get_position="[lon, lat]",
        get_fill_color=POINT_COLOR,
        get_line_color="[255, 255, 255, 255]",
        stroked=True,
        get_line_width=3,
        get_radius=25,
        radius_min_pixels=5,
        radius_max_pixels=15,
        pickable=True,
    )
    layers_to_render.append(layer_points)

    st.sidebar.markdown("---")
    if st.sidebar.button("Calcular Ruta Óptima", type="primary"):
        with st.spinner("Calculando sobre la red..."):
            dist_matrix = network_distance_matrix(G, nodes)
            optimal_route, total_distance = solve_tsp(dist_matrix, depot_index=0)
            
            if optimal_route and total_distance < 9999999:
                st.session_state['optimal_route'] = optimal_route
                st.session_state['total_distance'] = total_distance
                st.session_state['solved'] = True
            else:
                st.sidebar.error("Inviable: Algunos puntos no están interconectados vialmente.")

    if st.session_state.get('solved', False):
        optimal_route = st.session_state['optimal_route']
        total_dist_m = st.session_state['total_distance']
        
        # MÉTRICAS
        st.sidebar.subheader("Resultados:")
        km = total_dist_m / 1000.0
        # Velocidad estimada 40 km/h en urbano = 40/60 km por min
        mins = (km / 40.0) * 60.0
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Distancia", f"{km:.2f} km")
        col2.metric("Tiempo Aprox.", f"{mins:.1f} min")
        
        # Orden de Paradas
        with st.sidebar.expander("Secuencia de Visita (Coords)", expanded=False):
            st.write(f"Orden Lógico: {' -> '.join(map(str, optimal_route))}")
            for step, idx in enumerate(optimal_route):
                if step == 0:
                    st.write(f"**Origen:** {coords[idx][0]:.5f}, {coords[idx][1]:.5f}")
                elif step == len(optimal_route) - 1:
                    st.write(f"**Destino (Retorno):** {coords[idx][0]:.5f}, {coords[idx][1]:.5f}")
                else:
                    st.write(f"{step}. Parada {idx}: {coords[idx][0]:.5f}, {coords[idx][1]:.5f}")

        # Geometría Fija
        df_path = construct_full_geometry(G, nodes, optimal_route)
        layer_path = pdk.Layer(
            "PathLayer",
            df_path,
            width_scale=2,
            width_min_pixels=4,
            width_max_pixels=8,
            get_color=PATH_COLOR,
            get_path="path",
            pickable=True,
            line_joint_rounded=True,
            line_cap_rounded=True
        )
        layers_to_render.append(layer_path)
        
        # ANIMACIÓN
        if st.sidebar.button("🚗 Empezar Animación"):
            # En vez de dibujar toda la línea, la revelamos nodo por nodo de forma animada
            full_path = df_path["path"].iloc[0]
            anim_path = []
            
            for pt in full_path:
                anim_path.append(pt)
                anim_df = pd.DataFrame({"path": [anim_path]})
                anim_layer = pdk.Layer(
                    "PathLayer",
                    anim_df,
                    width_scale=2,
                    width_min_pixels=6,
                    width_max_pixels=12,
                    get_color="[0, 255, 50, 255]", # Verde brillante en animación
                    get_path="path",
                    line_joint_rounded=True,
                    line_cap_rounded=True
                )
                
                # Renderizar cuadro
                chart_placeholder.pydeck_chart(pdk.Deck(
                    layers=[layer_buildings, layer_points, anim_layer],
                    initial_view_state=view_state,
                    map_style=MAP_STYLE
                ))
                time.sleep(0.05) # Velocidad del coche
            
            # Dejar renderizado en su color normal terminado
            st.sidebar.success("Animación Finalizada")


# Render estático predeterminado
chart_placeholder.pydeck_chart(pdk.Deck(
    layers=layers_to_render, 
    initial_view_state=view_state, 
    map_style=MAP_STYLE,
    tooltip={"html": "<b>{index}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
))
