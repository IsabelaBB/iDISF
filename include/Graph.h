/**
* Dynamic and Iterative Spanning Forest
* 
* @date September, 2019
*/
#ifndef Graph_H
#define Graph_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Utils.h"
#include "Image.h"
#include "Color.h"
#include <omp.h>

//=============================================================================
// Structures
//=============================================================================
/**
* Node 2D coordinates
*/
typedef struct
{
    int x, y;
} NodeCoords;

/**
* Adjacency relation between the nodes of the graph
*/ 
typedef struct
{
    int size;
    // An adjacent coordinate can be obtained by simply adding the variations
    // along the x-axis and y-axis (dx and dy, respectively). Example: the adjacent
    // in the left can be obtained if dx[i] = -1, and dy[i] = 0, for an i < size
    int *dx, *dy;
} NodeAdj;

/**
* Abstract representation of an optimum-path tree
*/
typedef struct
{
    int root_index, num_nodes, num_feats;
    float minDist_userSeed; 
    // For speeding purposes, it is preferable to have such vector for the summation
    // of the pixel's features. 
    float *sum_feat;

    // For speeding purposes, it is preferable to have such vector for the summation
    // of the grad's pixels.     
    float sum_grad;   // sum (grad)
    float sum_grad_2; // sum (grad^2)
} Tree;

/**
* Image Graph
*/
typedef struct
{
    int num_cols, num_rows, num_feats, num_nodes;
    // Each node whose index is i < num_nodes, contain num_feats of features, which
    // can be obtained through feats[i].
    float **feats;
} Graph;

//=============================================================================
// Bool Functions
//=============================================================================
/**
* Evaluates if the coordinates of a given NodeCoords object are within the 
* domains of the image graph
*/
bool areValidNodeCoords(int num_rows, int num_cols, NodeCoords coords);

//=============================================================================
// Int Functions
//=============================================================================
/**
* Converts the coordinates of a given NodeCoords object into an index. Warning!
* It does not verify if the coordinates given are valid!
*/
int getNodeIndex(int num_cols, NodeCoords coords);

//=============================================================================
// Double Functions
//=============================================================================
/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two feature vectors
* of same dimensionality
*/
double euclDistance(float *feat1, float *feat2, int num_feats);

/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two features
*/
double euclDistanceValues(float feat1, float feat2);

/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two NodeCoords
*/
double euclDistanceCoords(NodeCoords feat1, NodeCoords feat2);

/**
* Computes the L1 Norm (a.k.a. Taxicab Distance) between two feature vectors 
* of same dimensionality
*/
double taxicabDistance(float *feat1, float *feat2, int num_feats);

/*
Compute the path cost
*/
double calcPathCost(float *mean_feat_tree, float *feats, int num_feats, double cost_map, int num_nodes_tree, double grad_adj, double coef_variation_tree, double alpha, double c2, int function);

//=============================================================================
// NodeCoords Functions
//=============================================================================
/**
* Get the coordinates of the id-th adjacent pixel determined by the adjacency
* relation considered. Warning! It does not evaluate whether the id given is 
* valid
*/
NodeCoords getAdjacentNodeCoords(NodeAdj *adj_rel, NodeCoords coords, int id);

/**
* Gets the coordinates of a pixel at the given index in the image graph. Warning!
* It does note evaluate whether the index is valid!
*/
NodeCoords getNodeCoords(int num_cols, int index);

//=============================================================================
// Float* Functions
//=============================================================================
/**
* Computes the mean feature vector of a given tree
*/
float* meanTreeFeatVector(Tree *tree);

/**
* Computes the mean gradient of a given tree
*/
float meanTreeGradVector(Tree *tree);

//=============================================================================
// Double* Functions
//=============================================================================
/**
* Computes the image gradient of the graph. It performs a summation of the
* of the ;weighted differences between a center pixel, and its adjacents.
*/
double *computeGradient(Graph *graph, double *coef_variation_img);
//=============================================================================
// NodeAdj* Functions
//=============================================================================
/**
* Creates a 4-neighborhood adjacency relation
*/
NodeAdj *create4NeighAdj();

/**
* Creates an 8-neighborhood adjacency relation
*/
NodeAdj *create8NeighAdj();

//=============================================================================
// Graph* Functions
//=============================================================================
/**
* Creates an image graph given the image in parameter. It considers an 4-adjacency
* relation between the nodes, and converts the image's features (expecting RGB)
* into the L*a*b* colorspace.
*/
Graph *createGraph(Image *img);

//=============================================================================
// Tree* Functions
//=============================================================================
/**
* Creates an empty tree rooted at the given index. Warning! It does not add the
* index features into the tree!
*/
Tree *createTree(int root_index, int num_feats);


//=============================================================================
// Void Functions
//=============================================================================
/**
* Deallocates the memory reserved for the adjacency relation given in parameter
*/ 
void freeNodeAdj(NodeAdj **adj_rel);

/**
* Deallocates the memory reserved for the tree given in parameter
*/ 
void freeTree(Tree **tree);

/**
* Deallocates the memory reserved for the image graph given in parameter
*/ 
void freeGraph(Graph **graph);

/**
* Inserts a node into the given tree. Thus, the node's features are added to
* the tree's summation vector, and the tree's size is increased by one. Warning!
* It does not verify whether such node was already inserted in this, or any other
* tree!
*/ 
void insertNodeInTree(Graph *graph, int index, Tree **tree, double grad);

void removeNodeFromTree(Graph *graph, int index, Tree **tree, double grad);

float coefTreeVariation(Tree *tree);

void resetTree(Tree **tree, Graph *graph, int root_index, double grad, int num_feats);

#ifdef __cplusplus
}
#endif

#endif
