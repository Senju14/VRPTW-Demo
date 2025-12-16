import folium
from typing import List
from src.backend.data_loader import Customer


def create_map(depot: Customer, customers: List[Customer], routes: List[List[int]] | None = None) -> folium.Map:
    m = folium.Map(location=[depot.y, depot.x], zoom_start=12)

    folium.Marker(
        location=[depot.y, depot.x],
        popup="Depot (0)",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    for c in customers:
        folium.Marker(
            location=[c.y, c.x],
            popup=f"Customer {c.id}: Demand {c.demand}, TW [{c.ready_time}, {c.due_date}]",
            icon=folium.Icon(color="blue", icon="user"),
        ).add_to(m)

    if routes:
        colors = [
            "green",
            "purple",
            "orange",
            "darkred",
            "lightred",
            "beige",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "white",
            "pink",
            "lightblue",
            "lightgreen",
            "gray",
            "black",
            "lightgray",
        ]
        all_customers = [depot] + customers
        for i, route in enumerate(routes):
            if not route:
                continue
            color = colors[i % len(colors)]
            locs = [(all_customers[n].y, all_customers[n].x) for n in route]
            folium.PolyLine(locs, color=color, weight=5, opacity=0.8).add_to(m)

    return m


def get_map_html(m: folium.Map) -> str:
    return m._repr_html_()