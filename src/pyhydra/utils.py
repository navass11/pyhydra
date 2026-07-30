"""
Interactive utilities for Jupyter notebooks (ipyleaflet-based map widgets).

Optional dependency: ``pip install ipyleaflet ipywidgets``
"""


def interactive_map(zoom=2, center=(0, 0)):
    """
    Render an interactive ipyleaflet map in a Jupyter notebook.

    Draw a rectangle to select a bounding box or click a marker to select a
    point.  The function returns a live list that is populated as soon as the
    user finishes drawing; read ``coordinates_list[0]`` in the next cell.

    Args:
        zoom   : Initial zoom level (default 2).
        center : (lat, lon) tuple for the initial map centre (default (0, 0)).

    Returns:
        coordinates_list: list that will contain either
            - ``[N, W, S, E]`` for rectangle selections, or
            - ``{'type': 'point', 'value': [lat, lon]}`` for point selections.
    """
    try:
        from ipyleaflet import Map, DrawControl
        from ipywidgets import Output, VBox
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "interactive_map requires ipyleaflet and ipywidgets. "
            "Install them with: pip install ipyleaflet ipywidgets"
        ) from exc

    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)

    draw_control = DrawControl()
    draw_control.rectangle = {
        "shapeOptions": {"color": "#6bc2e5", "fillOpacity": 0.5}
    }
    draw_control.marker = {"shapeOptions": {"color": "#ff0000"}}
    draw_control.circle = {}
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circlemarker = {}

    output = Output()
    coordinates_list = []

    def handle_draw(self, action, geo_json):
        nonlocal coordinates_list
        with output:
            geometry_type = geo_json["geometry"]["type"]
            coordinates = geo_json["geometry"]["coordinates"]
            output.clear_output()

            if action == "created":
                if geometry_type == "Polygon":
                    north = coordinates[0][1][1]
                    west  = coordinates[0][0][0]
                    south = coordinates[0][0][1]
                    east  = coordinates[0][2][0]
                    area  = [north, west, south, east]
                    coordinates_list.append(area)
                    print(
                        f"Selected area:\n"
                        f"  North: {north}\n"
                        f"  South: {south}\n"
                        f"  West : {west}\n"
                        f"  East : {east}"
                    )
                elif geometry_type == "Point":
                    lat, lon = coordinates[1], coordinates[0]
                    coordinates_list.append({"type": "point", "value": [lat, lon]})
                    print(f"Selected point:\n  Latitude: {lat}\n  Longitude: {lon}")

    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)
    display(VBox([m, output]))
    return coordinates_list
