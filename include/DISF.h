/**
* Dynamic and Iterative Spanning Forest
* 
* @date September, 2019
*/
#ifndef DISF_H
#define DISF_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Graph.h"
#include "IntList.h"
#include "IntLabeledList.h"
#include "PrioQueue.h"
#include <omp.h>

//=============================================================================
// Image* Functions
//=============================================================================
/**
* Extracts the superpixels using the Dynamic and Iterative Spanning Forest 
* algorithm. Given an initial number of N_0 seeds, it removes iteratively the 
* most irrelevant ones, until the number N_f of superpixels is achieved. It is 
* possible to obtain the border map by simply creating an Image whose width and
* height are the same as the original image, but the number of channels is 1. 
* However, if no border map is desired, simply define the respective border image 
* object as NULL. Warning! It does not verify if N_0 >> N_f!
*/
Image *runLabeledDISF(Graph *graph, int n_0, int n_f, NodeCoords **coords_user_seeds, Image **border_img);

//=============================================================================
// IntList* Functions
//=============================================================================
/**
* Performs a grid sampling in the image graph, in order to achieve an approximate
* number of seeds (given in parameter), and returns a list of seed pixels indexes. 
* Please, be aware that the real number of seeds can be very different from expected.
*/
IntList *gridSamplingDISF(Graph *graph, int num_seeds);

/**
* Selects the seeds which generated the K most relevant superpixels, according to
* their area and gradient (defined by the tree-adjacency relation given), and returns
* their root pixel indexes on a list. Warning! It does not verify if the number to 
* maintain is greater than the current number of trees!
*/
IntList *selectSeedDISF(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain);


#ifdef __cplusplus
}
#endif

#endif
