from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd
import folium
import requests
from functools import lru_cache
import time
import polyline

special_postcodes = {
    'BD112BZ': (53.758755, -1.689026),
    'WA119TY': (53.476785, -2.666254),
    'DUBLIN': (53.3498, -6.2603)
}

@lru_cache(maxsize=1000)
def get_coordinates(postcode):
    if postcode in special_postcodes:
        return special_postcodes[postcode]
    try:
        response = requests.get(f"https://api.postcodes.io/postcodes/{postcode}")
        if response.status_code == 200:
            data = response.json()
            return (data['result']['latitude'], data['result']['longitude'])
        else:
            print(f"Failed to geocode {postcode}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error geocoding {postcode}: {e}")
        return None

def load_data():
    jobs_df = pd.read_csv("/content/enhanced_job_distances.csv")
    distance_df = pd.read_csv("/content/new_driving_distance_matrix_miles.csv", index_col=0)
    return jobs_df, distance_df

def create_data_model(distance_matrix, postcodes, depot_index):
    data = {}
    data["distance_matrix"] = distance_matrix
    data["num_vehicles"] = 1
    data["depot"] = depot_index
    data["postcodes"] = postcodes
    return data

def get_route(data, manager, routing, solution):
    route = []
    index = routing.Start(0)
    route_distance = 0
    while not routing.IsEnd(index):
        node = data["postcodes"][manager.IndexToNode(index)]
        route.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    node = data["postcodes"][manager.IndexToNode(index)]
    route.append(node)
    route_distance_miles = route_distance / 1000.0
    return route, route_distance_miles

def solve_vrp_for_driver(data, driver_name):
    try:
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,
            3000000,
            True,
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT
        )
        search_parameters.time_limit.seconds = 10
        search_parameters.log_search = False

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            route, distance = get_route(data, manager, routing, solution)
            print(f"\nDriver: {driver_name} - Distance: {distance:.2f} miles")
            return route
        else:
            print(f"No solution found for Driver {driver_name} within 10 seconds.")
            return None
    except Exception as e:
        print(f"Error solving VRP for {driver_name}: {e}")
        return None

def visualize_routes(driver_routes, get_coordinates_func):
    m = folium.Map(location=[54.0, -2.0], zoom_start=6)
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue',
             'darkgreen', 'cadetblue', 'pink']
    route_groups = {}

    for i, (driver, route) in enumerate(driver_routes.items()):
        if not route:
            continue
        color = colors[i % len(colors)]
        coords = [get_coordinates_func(pc) for pc in route]

        # Create feature group for this driver
        fg = folium.FeatureGroup(name=driver, show=False)
        route_groups[driver] = fg

        # Build route path
        path_coords = []
        previous_valid_coord = None
        for pc, coord in zip(route, coords):
            if coord is None:
                previous_valid_coord = None
                continue

            if previous_valid_coord is not None:
                try:
                    url = f"http://router.project-osrm.org/route/v1/driving/{previous_valid_coord[1]},{previous_valid_coord[0]};{coord[1]},{coord[0]}?overview=full"
                    response = requests.get(url)
                    data = response.json()
                    if response.status_code == 200 and data.get('code') == 'Ok':
                        decoded = polyline.decode(data['routes'][0]['geometry'])
                        path_coords.extend(decoded)
                    else:
                        path_coords.extend([previous_valid_coord, coord])
                except Exception as e:
                    print(f"Error getting route for {driver}: {e}")
                    path_coords.extend([previous_valid_coord, coord])
                time.sleep(1)

            previous_valid_coord = coord

        if path_coords:
            folium.PolyLine(
                locations=path_coords,
                color=color,
                weight=2.5,
                opacity=1,
                popup=driver
            ).add_to(fg)

        # Add markers with sequence numbers
        for idx, (pc, coord) in enumerate(zip(route, coords)):
            if coord is None:
                continue

            position = idx + 1  # 1-based numbering
            if idx == 0:
                # Start marker
                folium.Marker(
                    location=coord,
                    popup=f"{position}. Start: {pc}",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(fg)
            elif idx == len(route) - 1:
                # End marker if different from start
                if pc != route[0]:
                    folium.Marker(
                        location=coord,
                        popup=f"{position}. End: {pc}",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(fg)
            else:
                # Intermediate markers
                folium.Marker(
                    location=coord,
                    popup=f"{position}. {pc}",
                    icon=folium.Icon(color=color)
                ).add_to(fg)

        # Add to map but hide initially
        fg.add_to(m)

    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    m.save('final_road_routes_label.html')
    print("Map saved as 'driver_routes.html'")

def main():
    jobs_df, distance_df = load_data()
    distance_matrix = (distance_df.values * 1000).round().astype(int).tolist()
    all_postcodes = distance_df.index.tolist()
    drivers = jobs_df["DRIVER NAME"].unique()
    driver_routes = {}

    for driver in drivers:
        driver_jobs = jobs_df[jobs_df["DRIVER NAME"] == driver]
        collection_postcodes = driver_jobs["COLLECTION POST CODE"].dropna().unique()
        deliver_postcodes = driver_jobs["DELIVER POST CODE"].dropna().unique()
        driver_postcodes = list(set(collection_postcodes) | set(deliver_postcodes))
        valid_postcodes = [pc for pc in driver_postcodes if pc in all_postcodes]

        if len(valid_postcodes) < 1:
            print(f"Skipping {driver} - no valid postcodes")
            continue

        postcode_to_index = {pc: idx for idx, pc in enumerate(all_postcodes)}
        indices = [postcode_to_index[pc] for pc in valid_postcodes]
        cluster_matrix = [[distance_matrix[i][j] for j in indices] for i in indices]
        depot_postcode = driver_jobs["COLLECTION POST CODE"].iloc[0]
        depot_index = valid_postcodes.index(depot_postcode) if depot_postcode in valid_postcodes else 0
        data = create_data_model(cluster_matrix, valid_postcodes, depot_index)

        print(f"\nSolving for {driver}...")
        route = solve_vrp_for_driver(data, driver)
        if route:
            driver_routes[driver] = route

    visualize_routes(driver_routes, get_coordinates)

if __name__ == "__main__":
    main()