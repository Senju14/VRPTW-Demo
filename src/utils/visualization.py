

import folium
from folium import plugins
from typing import List, Tuple
from src.utils.data_loader import Customer


def create_map(depot: Customer, customers: List[Customer], routes: List[List[int]] = None) -> folium.Map:
 
    # Center map on depot
    m = folium.Map(location=[depot.y, depot.x], zoom_start=12)

    # Add depot
    folium.Marker(
        location=[depot.y, depot.x],
        popup=f"Depot (0)",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    # Add customers
    for cust in customers:
        folium.Marker(
            location=[cust.y, cust.x],
            popup=f"Customer {cust.id}: Demand {cust.demand}, TW [{cust.ready_time}, {cust.due_date}]",
            icon=folium.Icon(color='blue', icon='user')
        ).add_to(m)

    # Add routes
    if routes:
        colors = ['green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        all_customers = [depot] + customers
        for i, route in enumerate(routes):
            if not route:
                continue
            color = colors[i % len(colors)]
            locations = [(all_customers[node].y, all_customers[node].x) for node in route]
            folium.PolyLine(locations, color=color, weight=5, opacity=0.8).add_to(m)

    return m


def get_map_html(map_obj: folium.Map) -> str:
   
    return map_obj._repr_html_()