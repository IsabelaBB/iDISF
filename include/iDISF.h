/**
* Dynamic and Iterative Spanning Forest
* 
* @date September, 2019
*/
#ifndef iDISF_H
#define iDISF_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Graph.h"
#include "IntList.h"
#include "PrioQueue.h"
#include <omp.h>

//=============================================================================
// Image* Functions
//=============================================================================
Image *runiDISF(Graph *graph, int n_0, int n_f, Image **border_img, NodeCoords** coords_user_seeds, int num_user_seeds, int *marker_sizes, int function, int all_borders, double c1, double c2, int num_objmarkers);
Image *runiDISF_scribbles_rem(Graph *graph, int n_0, int iterations, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, /*int sampling_method,*/ int obj_markers);


//=============================================================================
// IntList* Functions
//=============================================================================
IntList *gridSampling(int num_cols, int num_rows, int *num_seeds, NodeCoords** coords_user_seeds, int num_user_seeds, int *marker_sizes, double *grad, int *scribbled_seeds, int *nf);
IntList *gridSampling_scribbles(int num_rows, int num_cols, int *n_0, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, double *grad, int *labels_map, int obj_markers);

IntList *selectKMostRelevantSeeds(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain, int num_user_seeds);
IntList *seedRemoval(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_markers, int num_objmarkers, int *new_labels_map, int *stop);

#ifdef __cplusplus
}
#endif

#endif
