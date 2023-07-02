import vsketch
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import pandas as pd
import networkx as nx
from collections import deque
from shapely import Polygon

from plotter_shapes.plotter_shapes import draw_filled_circle, draw_dashed_line


# Option for removing dots if cluster is too big
# Remove dots if too close to other in cluster (keep the one furthest from center? Add debug draw of deleted)
# Remove edges where angle is too small (difficult to get math right on this...)

# Option to not draw the outliers
# Option for variable star size
# Option for dotted lines
# Cool borders - look at old starchart maps for inspiration


# Main TODOs:
# - Remove nodes close together before clustering (x)
# - Add extra set of stars which will not be clustered (x)
# - Different size of stars (x)
# - Extra edges but not for triangles with bad circumference to area ratio (x)
# - Redo graph iteration code to actually deal with the leftovers correctly
# - Heuristic for ignoring sharp angles? Hard to tune but should be easy to compute the angle
# - Circles around some stars (largest)
# - Randomly draw dotted lines between close stars across constellations
# - Somehow make more than one extra edge not do bad stuff


class StarchartsSketch(vsketch.SketchClass):
    WIDTH = 21
    HEIGHT = 29.7
    
    debug = vsketch.Param(True)
    debug_draw_full_tri = vsketch.Param(True)
    
    mode = vsketch.Param("circle", choices=["rect", "circle"])
    border = vsketch.Param("simple", choices=["none", "simple"])
    
    N_stars = vsketch.Param(350, min_value=0)
    stars_no_cluster_frac  = vsketch.Param(0.5, min_value=0.0, max_value=1.0)

    inner_padding = vsketch.Param(0.5, min_value=0.0)
    border_padding = vsketch.Param(0.5, min_value=0.0)
    
    eps = vsketch.Param(1.0, min_value=0.0)
    
    star_size_a_mean = vsketch.Param(0.16, min_value=0.0)
    star_size_a_std = vsketch.Param(0.05, min_value=0.0)
    star_size_min = vsketch.Param(0.1, min_value=0.0)
    star_size_a_max = vsketch.Param(0.4, min_value=0.0)
    star_size_b_mean = vsketch.Param(0.1, min_value=0.0)
    star_size_b_std = vsketch.Param(0.04, min_value=0.0)
    star_size_b_max = vsketch.Param(0.2, min_value=0.0)
    
    star_distance_min = vsketch.Param(0.2, min_value=0.0)
    
    p_extra_edges_0 = vsketch.Param(0.7, min_value=0.0, max_value=1.0)
    p_extra_edges_1 = vsketch.Param(0.3, min_value=0.0, max_value=1.0)
    p_extra_edges_2 = vsketch.Param(0.0, min_value=0.0, max_value=1.0)
    
    area_perimeter_ratio_thresh = vsketch.Param(0.06, min_value=0.0)
    
    circle_largest = vsketch.Param(True)
    circle_largest_thresh = vsketch.Param(0.055, min_value=0.0)
    circle_largest_pad = vsketch.Param(0.07, min_value=0.0)
    
    rng = default_rng(123)
    
    def sample_star_positions(self, N_stars, mode="rect"):
        if mode == "rect":
            pos = self.rng.uniform(self.padding * np.ones(2), [self.WIDTH - self.padding,
                                                 self.HEIGHT - self.padding], (N_stars, 2))
        elif mode == "circle":
            angle = self.rng.uniform(0, 2 * np.pi, N_stars)
            radius = (0.5 * self.WIDTH - self.padding) * np.sqrt(self.rng.uniform(0, 1, N_stars))
            pos = np.array([0.5 * self.WIDTH + radius * np.cos(angle), 
                            0.5 * self.HEIGHT + radius * np.sin(angle)]).T
        return pos
    
    def remove_stars_too_close(self, pos_stars, distance_min):
        distances = np.triu(cdist(pos_stars, pos_stars))
        distances[distances == 0.0] = np.nan
        indices_below_thresh = np.argwhere(distances < distance_min)
        N_indices_below_thresh = len(indices_below_thresh)
        indices_to_remove = self.rng.integers(2, size=N_indices_below_thresh)
        stars_to_remove = indices_below_thresh[range(N_indices_below_thresh), indices_to_remove]
        pos_stars = np.delete(pos_stars, stars_to_remove, axis=0)
        return pos_stars
    
    def sample_star_radii(self, N_stars, k_split):
        radii_stars = np.zeros(N_stars)
        # radii_stars = self.rng.uniform(self.star_size_min, self.star_size_max, self.N_stars)
        # radii_stars = 1.0 / self.rng.uniform(self.star_size_min, self.star_size_max, self.N_stars)**2
        # radii_stars = 0.4 * 1.0 / np.clip(self.rng.normal(3.0, 0.7, size=self.N_stars), a_min=1.4, a_max=10.0)**1
        radii_stars[:k_split] = np.clip(self.rng.normal(self.star_size_a_mean, self.star_size_a_std, size=k_split), 
                                        a_min=self.star_size_min, a_max=self.star_size_a_max)
        radii_stars[k_split:] = np.clip(self.rng.normal(self.star_size_b_mean, self.star_size_b_std, size=(N_stars - k_split)), 
                                        a_min=self.star_size_min, a_max=self.star_size_b_max)
        radii_stars *= radii_stars
        return radii_stars
    
    def cluster_stars(self, pos_stars):
        # Cluster data:        
        clustering = DBSCAN(eps=self.eps, min_samples=3).fit(pos_stars)
        labels = clustering.labels_
        
        # Group by cluster label:
        df_stars = pd.DataFrame()
        df_stars["x"] = pos_stars[:,0]
        df_stars["y"] = pos_stars[:,1]
        df_stars["label"] = labels
        star_clusters = df_stars[df_stars["label"] != -1].groupby("label")
        return star_clusters
    
    def build_graph(self, pos, edges):
        graph = nx.Graph()
        
        for i, (x, y) in enumerate(pos):
            graph.add_node(i, x=x, y=y)
        
        for node_1, node_2, node_3 in edges:
            for n_1, n_2 in ((node_1, node_2), (node_2, node_3), (node_3, node_1)):
                if not graph.has_edge(n_1, n_2):
                    d_12 = np.linalg.norm(pos[n_1] - pos[n_2])
                    graph.add_edge(n_1, n_2, distance=d_12)
        
        return graph

    def draw_graph(self, graph, vsk, dashed=False, debug=False):
        if debug:
            for node, pos in graph.nodes(data=True):
                # vsk.stroke(graph.cluster_label+2)
                vsk.circle(pos["x"], pos["y"], radius=5e-2) 
        
        for u, v in graph.edges:
            x_u, y_u = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x_v, y_v = graph.nodes[v]["x"], graph.nodes[v]["y"]
            if dashed:
                vsk.sketch(draw_dashed_line(x_u, y_u, x_v, y_v))
            else:
                vsk.line(x_u, y_u, x_v, y_v)

    def draw_stars(self, vsk, pos, radii):
        for pos, radius in zip(pos, radii):
            vsk.sketch(draw_filled_circle(pos[0], pos[1], radius=radius, line_width=5e-3))
            
    def draw_rect_border(self, vsk):
        vsk.line(self.border_padding, self.border_padding, self.WIDTH - self.border_padding, self.border_padding)
        vsk.line(self.border_padding, self.HEIGHT - self.border_padding, self.WIDTH - self.border_padding, self.HEIGHT - self.border_padding)
        vsk.line(self.border_padding, self.border_padding, self.border_padding, self.HEIGHT - self.border_padding)
        vsk.line(self.WIDTH - self.border_padding, self.border_padding, self.WIDTH - self.border_padding, self.HEIGHT - self.border_padding)
    
    def draw_circle_border(self, vsk):
        vsk.circle(0.5 * self.WIDTH, 0.5 * self.HEIGHT, radius=(0.5 * self.WIDTH - self.border_padding))
    
    def draw_init(self, vsk):
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        self.padding = self.inner_padding + self.border_padding
        self.p_extra_edges = np.array([self.p_extra_edges_0, self.p_extra_edges_1, self.p_extra_edges_2])
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        self.draw_init(vsk)

        pos_stars = self.sample_star_positions(self.N_stars, self.mode)
        
        # Remove stars too close to eachother:
        pos_stars = self.remove_stars_too_close(pos_stars, self.star_distance_min)
        self.N_stars = len(pos_stars)
        
        k_cluster = int(self.N_stars * (1.0 - self.stars_no_cluster_frac))
        
        radii_stars = self.sample_star_radii(self.N_stars, k_cluster)
        
        star_clusters = self.cluster_stars(pos_stars[:k_cluster])
        
        ###
        
        for label, df_cluster in star_clusters:
            pos = df_cluster[["x", "y"]].to_numpy()
            N_points = len(pos)
            
            # Build graph:
            tri = Delaunay(pos)
            full_graph = self.build_graph(pos, tri.simplices)
            full_graph.cluster_label = label
            
            ### Iterate graph:
            
            # Start node:
            # start_node = self.rng.integers(0, N_points)  # TODO: option for this
            cluster_mean = np.mean(pos, axis=0)  # Should be changed out with centroid of the hull around the cluster. Could be done by a union of all the triangles.
            start_node = np.argmax(np.linalg.norm(pos - cluster_mean, axis=1))
            
            reduced_graph = nx.Graph()
            
            queue = deque([start_node])
            valid_nodes = np.ones(N_points, dtype=bool)
            while len(queue) > 0:
                node = queue.popleft()
                reduced_graph.add_nodes_from([(node, full_graph.nodes[node])])
                
                valid_nodes[node] = False
                valid_edges = [x for x in full_graph[node].items() if valid_nodes[x[0]]]
                # print(f"\nPopped node {node}")
                # print(f"Valid nodes (all): {valid_nodes}")
                # print(f"Valid edges: {valid_edges}")
                
                if len(valid_edges) > 0:
                    # Pick closest neighbor node:
                    closest_neighbor, attributes = min(valid_edges, key=lambda edge: edge[1]["distance"])
                    queue.append(closest_neighbor)
                    
                    # Add edge to new graph:
                    reduced_graph.add_edges_from([(node, closest_neighbor, attributes)])
                    # print(f"Closest neighbor: {closest_neighbor}")
                    # print(f"Reduced graph: {reduced_graph}")
            
            # Randomly go through remaining nodes:
            # TODO: update this to continue to iterate... Requires some refactoring of code.
            while np.any(valid_nodes):
                nodes_left_to_visit = np.where(valid_nodes)[0]
                node = self.rng.choice(nodes_left_to_visit)
                reduced_graph.add_nodes_from([(node, full_graph.nodes[node])])
                valid_nodes[node] = False
                edges = [x for x in full_graph[node].items()]
                closest_neighbor, attributes = min(edges, key=lambda edge: edge[1]["distance"])
                reduced_graph.add_edges_from([(node, closest_neighbor, attributes)])


            # Pick how many extra edges to add:
            edges_left = len(full_graph.edges) - len(reduced_graph.edges)
            # print("Edges left:", edges_left)
            num_extra_edges = self.rng.choice(range(len(self.p_extra_edges)), p=self.p_extra_edges)
            num_extra_edges = np.min((num_extra_edges, edges_left))
            # print("Extra edges:", num_extra_edges)
            
            if num_extra_edges > 0:
                graph_difference = nx.difference(full_graph, reduced_graph)  # get graph with edges not yet added

                # Remove edges which have bad area / circumference ratio:
                for node_from, node_to in graph_difference.edges:  # for every edge not added to reduced graph
                    # curr_distance = full_graph.edges[node_from, node_to]["distance"]  # distance of current edge
                    path = nx.shortest_path(reduced_graph, node_from, node_to, weight="distance")  # get shortest path
                    
                    # Create polygon and get area and circumference:
                    poly_points = [(full_graph.nodes[n]["x"], full_graph.nodes[n]["y"]) for n in path]
                    poly = Polygon(poly_points)
                    area_perimeter_ratio = poly.area / poly.length
                    if area_perimeter_ratio < self.area_perimeter_ratio_thresh:
                        graph_difference.remove_edge(node_from, node_to)
                
                # Randomly add edges:
                edges_left = len(graph_difference.edges)
                num_extra_edges = np.min((num_extra_edges, edges_left))
                if num_extra_edges > 0:
                    extra_edges = self.rng.choice(graph_difference.edges, num_extra_edges)
                    for node_from, node_to in extra_edges:
                        attributes = full_graph.edges[node_from, node_to]
                        reduced_graph.add_edges_from([(node_from, node_to, attributes)])


            # Draw full graph:
            if self.debug_draw_full_tri:
                vsk.stroke(2)
                self.draw_graph(full_graph, vsk, dashed=True)
            
            # Other debug draws:
            if self.debug:
                vsk.circle(cluster_mean[0], cluster_mean[1], radius=1e-2)
                vsk.circle(pos[start_node,0], pos[start_node,1], radius=1e-1)
                
            # Draw reduced graph:            
            vsk.stroke(1)
            self.draw_graph(reduced_graph, vsk)
        
        ###
        
        # Draw border:
        vsk.stroke(1)
        if self.border == "simple":
            if self.mode == "rect":
                self.draw_rect_border(vsk)
            elif self.mode == "circle":
                self.draw_circle_border(vsk)
        
        # Draw stars:                
        self.draw_stars(vsk, pos_stars, radii_stars)
        
        if self.circle_largest:
            indices = np.argwhere(radii_stars > self.circle_largest_thresh)
            for pos, radius in zip(pos_stars[indices].squeeze(), radii_stars[indices].squeeze()):
                vsk.circle(pos[0], pos[1], radius=radius+self.circle_largest_pad)
        
        # Old stuff to draw outliers:   
        # for label, pos in zip(labels, pos_stars):
        #     if label == -1:
        #         vsk.line(pos[0], pos[1] - 0.1, pos[0], pos[1] + 0.1)
        #         vsk.line(pos[0] - 0.1, pos[1], pos[0] + 0.1, pos[1])
        #     # else:
            #     vsk.stroke(label+2)
            #     vsk.circle(pos[0], pos[1], radius=1e-1)
        
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    StarchartsSketch.display()
