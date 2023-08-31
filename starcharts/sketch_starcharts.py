import vsketch
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import pandas as pd
import networkx as nx
import heapq
from shapely import Polygon

from plotter_shapes.plotter_shapes import draw_filled_circle, draw_dashed_line, draw_cross, draw_asterix, rotate_and_draw_sketch


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
# - Redo graph iteration code to actually deal with the leftovers correctly (x)
# - Circles around some stars (largest) (x)
# - Randomly draw dotted lines between close stars across constellations (x)
# - 3-way or cross stars with random rotation? For a few select stars instead of filled circle? Why not try it, fast to do
# - Cool borders - look at old starchart maps for inspiration
# Honestly after the two points above I think it is done
# - Heuristic for ignoring sharp angles? Hard to tune but should be easy to compute the angle
# - Somehow make more than one extra edge not do bad stuff


class StarchartsSketch(vsketch.SketchClass):
    WIDTH = 21
    HEIGHT = 29.7
    
    debug = vsketch.Param(True)
    debug_draw_full_tri = vsketch.Param(True)
    
    mode = vsketch.Param("circle", choices=["rect", "circle"])
    border = vsketch.Param("simple", choices=["none", "simple"])
    
    N_stars = vsketch.Param(500, min_value=0)
    stars_no_cluster_frac  = vsketch.Param(0.6, min_value=0.0, max_value=1.0)

    inner_padding = vsketch.Param(0.3, min_value=0.0)
    border_padding = vsketch.Param(0.9, min_value=0.0)
    
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
    
    line_largest = vsketch.Param(True)
    line_largest_frac = vsketch.Param(0.4, min_value=0.0)
    line_largest_dash_size = vsketch.Param(0.25, min_value=0.0)
    line_largest_dash_factor = vsketch.Param(0.4, min_value=0.0, max_value=1.0)
    
    p_star_dot = vsketch.Param(0.8, min_value=0.0, max_value=1.0)
    p_star_cross = vsketch.Param(0.1, min_value=0.0, max_value=1.0)
    p_star_ast = vsketch.Param(0.1, min_value=0.0, max_value=1.0)
    
    cross_radius_gain = vsketch.Param(3.0, min_value=0.0)
    ast_radius_gain = vsketch.Param(3.0, min_value=0.0)
    
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
    
    @staticmethod
    def build_graph(pos):
        tri = Delaunay(pos)
        edges = tri.simplices
        
        graph = nx.Graph()
        
        for i, (x, y) in enumerate(pos):
            graph.add_node(i, x=x, y=y)
        
        for node_1, node_2, node_3 in edges:
            for n_1, n_2 in ((node_1, node_2), (node_2, node_3), (node_3, node_1)):
                if not graph.has_edge(n_1, n_2):
                    d_12 = np.linalg.norm(pos[n_1] - pos[n_2])
                    graph.add_edge(n_1, n_2, distance=d_12)
        
        return graph

    @staticmethod
    def build_reduced_graph(full_graph, start_node):
        N_nodes = len(full_graph.nodes)
        
        reduced_graph = nx.Graph()

        valid_nodes = np.ones(N_nodes, dtype=bool)  # Boolean array to keep track of visited nodes
        valid_edges_heap = []
        
        def visit_node(node):
            # vsk.text(str(i_nodes), full_graph.nodes[node]["x"] + 2e-1, full_graph.nodes[node]["y"] - 2e-1, align="center", size=0.2)
            
            valid_nodes[node] = False
            reduced_graph.add_nodes_from([(node, full_graph.nodes[node])])
            
            # Push new possible edges:
            new_edges = [x for x in full_graph[node].items() if valid_nodes[x[0]]]
            for to_node, attributes in new_edges:
                heap_edge_tuple = (attributes["distance"], node, to_node, attributes)
                heapq.heappush(valid_edges_heap, heap_edge_tuple)
            
        visit_node(start_node)
        
        while np.any(valid_nodes):
            _, from_node, to_node, attributes = heapq.heappop(valid_edges_heap)
            if valid_nodes[to_node]:
                visit_node(to_node)  # visit node and update heap structure with new edges
                reduced_graph.add_edges_from([(from_node, to_node, attributes)])
        
        return reduced_graph

    def add_extra_adges(self, full_graph, reduced_graph, num_extra_edges, area_perimeter_ratio_thresh):
        graph_difference = nx.difference(full_graph, reduced_graph)  # get graph with edges not yet added

        # Remove edges which have bad area / circumference ratio:
        for node_from, node_to in graph_difference.edges:  # for every edge not added to reduced graph
            # curr_distance = full_graph.edges[node_from, node_to]["distance"]  # distance of current edge
            path = nx.shortest_path(reduced_graph, node_from, node_to, weight="distance")  # get shortest path
            
            # Create polygon and get area and circumference:
            poly_points = [(full_graph.nodes[n]["x"], full_graph.nodes[n]["y"]) for n in path]
            poly = Polygon(poly_points)
            area_perimeter_ratio = poly.area / poly.length
            if area_perimeter_ratio < area_perimeter_ratio_thresh:
                graph_difference.remove_edge(node_from, node_to)
        
        # Randomly add edges:
        edges_left = len(graph_difference.edges)
        num_extra_edges = np.min((num_extra_edges, edges_left))
        if num_extra_edges > 0:
            extra_edges = self.rng.choice(graph_difference.edges, num_extra_edges)
            for node_from, node_to in extra_edges:
                attributes = full_graph.edges[node_from, node_to]
                reduced_graph.add_edges_from([(node_from, node_to, attributes)]) 
                    
        return reduced_graph

    def draw_graph(self, graph, vsk, dashed=False, debug=False):
        # if debug:
            # for node, pos in graph.nodes(data=True):
                # vsk.stroke(graph.cluster_label+2)
                # vsk.circle(pos["x"], pos["y"], radius=5e-2) 
                # vsk.text(str(node), pos["x"] + 2e-1, pos["y"] - 2e-1, align="center", size=0.2)
        
        for u, v in graph.edges:
            x_u, y_u = graph.nodes[u]["x"], graph.nodes[u]["y"]
            x_v, y_v = graph.nodes[v]["x"], graph.nodes[v]["y"]
            if dashed:
                vsk.sketch(draw_dashed_line(x_u, y_u, x_v, y_v))
            else:
                vsk.line(x_u, y_u, x_v, y_v)

    def draw_stars(self, vsk, pos, radii, type="dot"):
        assert type in ("dot", "cross", "ast")
        
        for pos, radius in zip(pos, radii):
            if type == "dot":
                vsk.sketch(draw_filled_circle(pos[0], pos[1], radius=radius, line_width=5e-3))
            elif type == "cross":
                sketch = draw_cross(0, 0, size=radius)
                rotate_and_draw_sketch(vsk, sketch, pos[0], pos[1], angle=self.rng.uniform(0, np.pi))
            elif type == "ast":
                sketch = draw_asterix(0, 0, size=radius)
                rotate_and_draw_sketch(vsk, sketch, pos[0], pos[1], angle=self.rng.uniform(0, np.pi))
            
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
        # self.p_star_shape = np.array([self.p_star_dot, self.p_star_cross, self.p_star_ast])
    
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
            # Build graph:
            pos = df_cluster[["x", "y"]].to_numpy()
            full_graph = self.build_graph(pos)
            full_graph.cluster_label = label
            
            
            # Decide start node:
            # start_node = self.rng.integers(0, N_points)  # TODO: option for this
            cluster_mean = np.mean(pos, axis=0)  # TODO: should be changed out with centroid of the hull around the cluster. Could be done by a union of all the triangles.
            start_node = np.argmax(np.linalg.norm(pos - cluster_mean, axis=1))
            
            # Iterate graph to build a reduced version:
            reduced_graph = self.build_reduced_graph(full_graph, start_node)


            # Pick how many extra edges to add:
            edges_left = len(full_graph.edges) - len(reduced_graph.edges)
            num_extra_edges = self.rng.choice(range(len(self.p_extra_edges)), p=self.p_extra_edges)
            num_extra_edges = np.min((num_extra_edges, edges_left))
            # print("Edges left:", edges_left)
            # print("Extra edges:", num_extra_edges)
            
            if num_extra_edges > 0:
                reduced_graph = self.add_extra_adges(full_graph, reduced_graph, num_extra_edges, self.area_perimeter_ratio_thresh)


            # Draw full graph:
            if self.debug_draw_full_tri:
                vsk.stroke(2)
                self.draw_graph(full_graph, vsk, dashed=True, debug=self.debug)
            
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
        N_stars_no_cluster = self.N_stars - k_cluster   
        N_cross_stars = int(np.round(self.p_star_cross * N_stars_no_cluster))
        N_ast_stars = int(np.round(self.p_star_ast * N_stars_no_cluster))
        self.draw_stars(vsk, pos_stars[:-(N_ast_stars + N_cross_stars)], radii_stars, type="dot")
        self.draw_stars(vsk, pos_stars[-(N_ast_stars + N_cross_stars):-N_ast_stars], 
                        self.cross_radius_gain * radii_stars, type="cross")
        self.draw_stars(vsk, pos_stars[-N_ast_stars:], self.ast_radius_gain * radii_stars, type="ast")

        # Circle around largest stars:
        indices_largest = np.argwhere(radii_stars > self.circle_largest_thresh)
        if self.circle_largest:
            for pos, radius in zip(pos_stars[indices_largest].squeeze(), radii_stars[indices_largest].squeeze()):
                vsk.circle(pos[0], pos[1], radius=radius+self.circle_largest_pad)

        # Line between largest stars:
        if self.line_largest:
            graph_largest = self.build_graph(pos_stars[indices_largest].squeeze())
            N_edge_picks = int(self.line_largest_frac * len(graph_largest.edges))
            edge_picks = self.rng.choice(graph_largest.edges, N_edge_picks)
            for n_from, n_to in edge_picks:
                vsk.sketch(draw_dashed_line(graph_largest.nodes[n_from]["x"], graph_largest.nodes[n_from]["y"],
                                            graph_largest.nodes[n_to]["x"], graph_largest.nodes[n_to]["y"], 
                                            dash_size=self.line_largest_dash_size, factor=self.line_largest_dash_factor))
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    StarchartsSketch.display()
