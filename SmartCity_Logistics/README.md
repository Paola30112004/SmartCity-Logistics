# Smart City Logistics 3D - Optimización de Rutas

Aplicación web interactiva desarrollada con **Streamlit** y **Pydeck** para la optimización de rutas de logística y transporte en un entorno geográfico semi-3D. Utiliza datos de calles de **OpenStreetMap** (mediante OSMnx/NetworkX) y resuelve el Problema del Viajante (TSP) evaluando combinatorias usando el motor especializado de **Google OR-Tools**.

---

## 📍 Ubicación del Proyecto

**Ruta de Instalación:**
```
C:\Users\Dell\OneDrive\Desktop\ProyectoSimulacion\SmartCity_logistics
```

---

## 🛠 Instalación y Configuración

Los siguientes comandos deben ser ejecutados desde la terminal del sistema (`PowerShell` o `CMD`) dentro de la ruta principal del proyecto.

### 1. Activar el Entorno Virtual (Recomendado)
El proyecto cuenta con un entorno virtual que localiza sus librerías. Puedes activarlo de la siguiente forma:
```bash
# En Windows:
.\venv\Scripts\activate
```

### 2. Instalar Dependencias
Asegúrate de tener instaladas todas las librerías necesarias con el gestor `pip`. Ejecuta el siguiente comando:
```bash
pip install streamlit pydeck osmnx numpy pandas networkx fpdf ortools PyPDF2
```

---

## 🚀 ¿Cómo Correr el Proyecto?

Una vez que todas las dependencias estén satisfechas, levanta el servidor local de **Streamlit** ejecutando:

```bash
python -m streamlit run app.py
```
> El script levantará un proceso en tu navegador predeterminado y abrirá automáticamente en `http://localhost:8501`.

---

## 📚 Documentación del Código Principal

El software está dividido en estructuras precisas responsables de la visualización, cálculos e I/O. Los módulos fundamentales son:

### 1. `app.py`
Archivo de entrada de la aplicación que administra toda la Interfaz de Usuario (UI) y flujos dinámicos:
- **Carga de Mapas y Edificios (`fetch_street_network`, `fetch_building_data`):** Extrae un grafo dirigible desde OSMnx partiendo del sector "Alta Vista, Ciudad Guayana". Genera altura aleatoria de polígonos habitacionales para propiciar perspectiva 3D a través de `pydeck.Layer`.
- **Generación de Paradas (`generate_and_snap_points`):** Permite fabricar paradas totalmente aleatorias pero aplicándolas (snapping) inteligentemente sobre los nodos asfálticos válidos más adyacentes de la red (para que las paradas no sean generadas sobre techos o lagos).
- **Motor Logístico (`network_distance_matrix`, `solve_tsp`):** Lógica encargada de correr un bucle de Dijkstra para trazar la distancia real rodando sobre autopistas entre todos los N puntos, entregando esta matriz matemática a `pywrapcp` (el constraint solver de OR-Tools) el cual retornará la combinatoria de visitas más baratas `PATH_CHEAPEST_ARC`.
- **Vista de PyDeck y Animación:** Renderizado de las torres, paradas, e interpolación polilinea para dibujar las calles correctas. Implementa un motor de fotograma en tiempo de repolarización en base al input de la velocidad del tipo de vehículo simulando física de avance con latencias dictadas por tiempo y distancias.
- **Exportación Automática (`generate_pdf_report`):** Una vez concluida la carrera, formatea un despacho métrico usando `fpdf` con distancias, combustible derivado de km, y tiempos para descargar localmente en PDF o Coordenadas a CSV.

### 2. `venv/tsp_solver.py`
Módulo enfocado estrictamente en resolución programática algorítmica y vectorización matemática:
- **Haversine Numpy Vectorizado (`haversine_distance_matrix`):** Una sub-implementación orientada a calcular matrices de distancia geográfica de vuelos a través de tensores multidimensionales con NumPy (`np.radians`, `np.newaxis`) suprimiendo por completo los loops `for`, otorgando hiper escalabilidad y velocidad computacional en arreglos considerables.
- **`solve_tsp`:** La firma del método matriz de la API de Google OR-Tools libre de lógica gráfica.
- **Testing Script:** Incorpora un final bloque de pruebas de script global local `if __name__ == "__main__":` para comprobar y validar cálculos asimulados por consola si no se desea apelar a toda la estructura gráfica robusta de Streamlit.

### 3. `read_pdf.py`
Un script auxiliar para recuperar información teórica del caso de estudio:
- Se conecta mediante acceso binario y extrae capa-por-capa un PDF documental adyacente en el Desktop del sistema procesándolo con `PyPDF2`.
- Cuenta con un bloque _"lazy-installer"_ mediante el proceso de librerias `subprocess.check_call` que fuerza vía proceso la instalación paralela del módulo de lectura si nota que sufre una irregularidad del `ImportError` impidiendo caída del programa.
- Retorna el dump de todo extracto en `pdf_content.txt`.
