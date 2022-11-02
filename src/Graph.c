#include "Graph.h"

//=============================================================================
// Auxiliary Functions
//=============================================================================
inline double calcPathCost(float *mean_feat_tree, float *feats, int num_feats, double cost_map, int num_nodes_tree, double grad_adj, double coef_variation_tree, double alpha, double c2, int function)
{
    double arc_cost, path_cost, diff_grad, beta;

    arc_cost = euclDistance(mean_feat_tree, feats, num_feats);

    if (function == 1) // dynamic color distance (same as DISF)
        path_cost = MAX(cost_map, arc_cost);
    else
    {
        if (function == 2)
        { // not normalization 
            beta = MAX(MAX(1, alpha * c2), coef_variation_tree);
            diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
            path_cost = MAX(cost_map, diff_grad);
        }
        else
        {
            if (function == 3)
            { 
                // sum gradient not norm
                beta = MAX(MAX(1, alpha * c2), coef_variation_tree);
                diff_grad = arc_cost + (pow(grad_adj, (1 / alpha)) * (1 / beta)); 
                path_cost = MAX(cost_map, diff_grad);
                
            }
            else
            {
                if (function == 4)
                { // cv normalization
                    beta = MAX(MAX(1, alpha * c2), coef_variation_tree / (float)num_nodes_tree); 
                    diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
                    path_cost = MAX(cost_map, diff_grad);
                }
                else
                {
                    if (function == 5)
                    { 
                        // beta normalization
                        beta = MAX(MAX(1, alpha * c2), coef_variation_tree) / (float)num_nodes_tree;
                        diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
                        path_cost = MAX(cost_map, diff_grad);
                    }
                    else
                    { // sum gradient beta norm
                        beta = MAX(MAX(1, alpha * c2), coef_variation_tree) / (float)num_nodes_tree;
                        diff_grad = arc_cost + pow(grad_adj, (1 / alpha)) * (1 / beta);
                        path_cost = MAX(cost_map, diff_grad);
                    }
                }
            }
        }
    }

    return path_cost;
}

inline float coefTreeVariation(Tree *tree)
{
    double tree_variance;
    double grad_mean, coef_variation_tree;

    tree_variance = (tree->sum_grad_2 / (double)tree->num_nodes) - ((tree->sum_grad * tree->sum_grad) / (((double)tree->num_nodes) * (double)tree->num_nodes));
    grad_mean = tree->sum_grad / (double)tree->num_nodes;
    coef_variation_tree = sqrt(MAX(0, tree_variance)) / MAX(0.00001, grad_mean);

    return coef_variation_tree;
}


//=============================================================================
// Constructors & Deconstructors
//=============================================================================
NodeAdj *create4NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = allocMem(1, sizeof(NodeAdj));

    adj_rel->size = 4;
    adj_rel->dx = allocMem(4, sizeof(int));
    adj_rel->dy = allocMem(4, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom

    return adj_rel;
}

NodeAdj *create8NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = allocMem(1, sizeof(NodeAdj));

    adj_rel->size = 8;
    adj_rel->dx = allocMem(8, sizeof(int));
    adj_rel->dy = allocMem(8, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Center-Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Center-Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top-Center
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom-Center

    adj_rel->dx[4] = -1;
    adj_rel->dy[4] = 1; // Bottom-Left
    adj_rel->dx[5] = 1;
    adj_rel->dy[5] = -1; // Top-Right

    adj_rel->dx[6] = -1;
    adj_rel->dy[6] = -1; // Top-Left
    adj_rel->dx[7] = 1;
    adj_rel->dy[7] = 1; // Bottom-Right

    return adj_rel;
}

Graph *createGraph(Image *img)
{
    int normval, num_nodes, num_channels, i;
    //NodeAdj *adj_rel;
    Graph *graph;

    normval = getNormValue(img);

    graph = allocMem(1, sizeof(Graph));

    graph->num_cols = img->num_cols;
    graph->num_rows = img->num_rows;
    graph->num_feats = 3; // L*a*b cspace
    num_nodes = graph->num_nodes = img->num_pixels;
    num_channels = img->num_channels;

    graph->feats = allocMem(num_nodes, sizeof(float *));

/*#pragma omp parallel for private(i)                \
    firstprivate(num_nodes, num_channels, normval) \
        shared(graph, img)*/
    for (i = 0; i < num_nodes; i++)
    {
        if (num_channels <= 2) // Grayscale w/ w/o alpha
            graph->feats[i] = convertGrayToLab(img->val[i], normval);
        else // sRGB
            graph->feats[i] = convertsRGBToLab(img->val[i], normval);

        /*
        float *srgb;
        srgb = allocMem(3, sizeof(float));
        srgb[0] = srgb[1] = srgb[2] = (float)img->val[i][0];
        graph->feats[i] = srgb;
        NodeCoords coords = getNodeCoords(num_cols, i);
       printf("%d;%d -> %f;  ", coords.x, coords.y, srgb[2]);
       */
    }
    /*
    adj_rel = create8NeighAdj();

    // Smoothing
    for (int i = 0; i < graph->num_nodes; i++)
    {
        NodeCoords node_coords;

        node_coords = getNodeCoords(num_cols, i);

        for (int j = 0; j < graph->num_feats; j++)
        {
            float smooth_val;

            smooth_val = graph->feats[i][j] * GAUSSIAN_3x3[0];

            // For each adjacent node
            for (int k = 0; k < adj_rel->size; k++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, k);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index;

                    adj_index = getNodeIndex(num_cols, adj_coords);

                    smooth_val += graph->feats[adj_index][j] * GAUSSIAN_3x3[k];
                }
            }

            graph->feats[i][j] = smooth_val;
        }
    }

    freeNodeAdj(&adj_rel);
    */
    return graph;
}

Tree *createTree(int root_index, int num_feats)
{
    Tree *tree;

    tree = allocMem(1, sizeof(Tree));

    tree->root_index = root_index;
    tree->num_nodes = 0;
    tree->num_feats = num_feats;
    tree->sum_grad = 0;
    tree->sum_grad_2 = 0;
    tree->minDist_userSeed = INFINITY;

    tree->sum_feat = allocMem(num_feats, sizeof(float));

    return tree;
}

void resetTree(Tree **tree, Graph *graph, int root_index, double grad, int num_feats)
{
    (*tree)->root_index = root_index;
    (*tree)->num_nodes = 1;
    (*tree)->num_feats = num_feats;
    (*tree)->sum_grad = grad;
    (*tree)->sum_grad_2 = grad*grad;
    (*tree)->minDist_userSeed = INFINITY;

    if(root_index != -1){
        for (int i = 0; i < num_feats; i++)
            (*tree)->sum_feat[i] = graph->feats[root_index][i];
    }
}

void freeNodeAdj(NodeAdj **adj_rel)
{
    if (*adj_rel != NULL)
    {
        NodeAdj *tmp;

        tmp = *adj_rel;

        freeMem(tmp->dx);
        freeMem(tmp->dy);
        freeMem(tmp);

        *adj_rel = NULL;
    }
}

void freeGraph(Graph **graph)
{
    if (*graph != NULL)
    {
        Graph *tmp;
        int i, num_nodes;

        tmp = *graph;
        num_nodes = tmp->num_nodes;

#pragma omp parallel for private(i) \
    firstprivate(num_nodes)         \
        shared(tmp)
        for (i = 0; i < num_nodes; i++)
            freeMem(tmp->feats[i]);
        freeMem(tmp->feats);
        freeMem(tmp);

        *graph = NULL;
    }
}

void freeTree(Tree **tree)
{
    if (*tree != NULL)
    {
        Tree *tmp;

        tmp = *tree;

        freeMem(tmp->sum_feat);
        freeMem(tmp);

        *tree = NULL;
    }
}

//=============================================================================
// Bool Functions
//=============================================================================
inline bool areValidNodeCoords(int num_rows, int num_cols, NodeCoords coords)
{
    return (coords.x >= 0 && coords.x < num_cols) &&
           (coords.y >= 0 && coords.y < num_rows);
}

//=============================================================================
// Int Functions
//=============================================================================
inline int getNodeIndex(int num_cols, NodeCoords coords)
{
    return coords.y * num_cols + coords.x;
}

//=============================================================================
// Double Functions
//=============================================================================
inline double euclDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for (int i = 0; i < num_feats; i++)
        dist += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
    dist = sqrtf(dist);

    return dist;
}

inline double taxicabDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for (int i = 0; i < num_feats; i++)
        dist += fabs(feat1[i] - feat2[i]);

    return dist;
}

inline double euclDistanceCoords(NodeCoords feat1, NodeCoords feat2)
{
    double dist;

    dist = 0;

    dist += ((float)feat1.x - (float)feat2.x) * ((float)feat1.x - (float)feat2.x);
    dist += ((float)feat1.y - (float)feat2.y) * ((float)feat1.y - (float)feat2.y);
    dist = sqrtf(dist);

    return dist;
}

//=============================================================================
// NodeCoords Functions
//=============================================================================
inline NodeCoords getAdjacentNodeCoords(NodeAdj *adj_rel, NodeCoords coords, int id)
{
    NodeCoords adj_coords;

    adj_coords.x = coords.x + adj_rel->dx[id];
    adj_coords.y = coords.y + adj_rel->dy[id];

    return adj_coords;
}

inline NodeCoords getNodeCoords(int num_cols, int index)
{
    NodeCoords coords;

    coords.x = index % num_cols;
    coords.y = index / num_cols;

    return coords;
}

//=============================================================================
// Float* Functions
//=============================================================================
inline float *meanTreeFeatVector(Tree *tree)
{
    float *mean_feat;

    mean_feat = allocMem(tree->num_feats, sizeof(float));

    for (int i = 0; i < tree->num_feats; i++)
        mean_feat[i] = tree->sum_feat[i] / (float)tree->num_nodes;

    return mean_feat;
}

//=============================================================================
// Double* Functions
//=============================================================================
double *computeGradient(Graph *graph, double *coef_variation_img)
{
    float max_adj_dist, sum_weight;
    float *dist_weight;
    double *grad;
    NodeAdj *adj_rel;
    double sum_grad = 0, sum_grad_2 = 0;
    double variance, mean;
    int num_cols, num_rows, num_nodes, num_feats;
    //double max_grad, min_grad;

    float div, *feats, *adj_feats;

    int i, j, rel_size, adj_index;
    NodeCoords coords, adj_coords;
    double dist;

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    grad = allocMem(num_nodes, sizeof(double));
    adj_rel = create8NeighAdj();

    rel_size = adj_rel->size;
    max_adj_dist = sqrtf(2); // Diagonal distance for 8-neighborhood
    dist_weight = allocMem(adj_rel->size, sizeof(float));
    sum_weight = 0;

    /* private : Cláusula que define que as variáveis definidas em list são duplicadas em cada
thread e o seu acesso passa a ser local (privado) em cada thread. O valor inicial das variáveis
privadas é indefinido (não é iniciado) e o valor final das variáveis originais (depois da região
paralela) também é indefinido.*/

    // firstprivate : Variáveis privadas que são inicializadas quando o código paralelo é iniciado

    /* shared : Cláusula que define sobre as variáveis definidas em list são partilhadas por todos
os threads ficando à responsabilidade do programador garantir o seu correto manuseamento.
Por omissão, as variáveis para as quais não é definido qualquer tipo são consideradas variáveis
partilhadas. */

    //max_grad = -1;
    //min_grad = 20000;

/*#pragma omp parallel for private(i, div) \
    firstprivate(max_adj_dist, rel_size) \
        shared(dist_weight, adj_rel)     \
            reduction(+                  \
                      : sum_weight)*/
    // Compute the inverse distance weights (closer --> higher; farther --> lower)
    for (i = 0; i < rel_size; i++)
    {
        // Distance between the adjacent and the center
        div = sqrtf(adj_rel->dx[i] * adj_rel->dx[i] + adj_rel->dy[i] * adj_rel->dy[i]);

        dist_weight[i] = max_adj_dist / div;
        sum_weight += dist_weight[i];
    }

/*#pragma omp parallel for private(i)    \
    firstprivate(rel_size, sum_weight) \
        shared(dist_weight)*/
    // Normalize values
    for (i = 0; i < rel_size; i++)
    {
        dist_weight[i] /= sum_weight;
    }

    // Compute the gradients
    for (i = 0; i < num_nodes; i++)
    {
        feats = graph->feats[i];
        coords = getNodeCoords(num_cols, i);

        // For each adjacent node
        for (j = 0; j < rel_size; j++)
        {
            adj_coords = getAdjacentNodeCoords(adj_rel, coords, j);

            if (areValidNodeCoords(num_rows, num_cols, adj_coords))
            {
                //adj_index = getNodeIndex(num_cols, adj_coords);
                adj_index = adj_coords.y * num_cols + adj_coords.x;
                adj_feats = graph->feats[adj_index];

                // Compute L1 Norm between center and adjacent
                dist = taxicabDistance(adj_feats, feats, num_feats);

                // Weight by its distance relevance
                grad[i] += dist * dist_weight[j];
            }
        }
        sum_grad += grad[i];
        sum_grad_2 += grad[i] * grad[i];
    }

    variance = (sum_grad_2 / (float)graph->num_nodes) - ((sum_grad * sum_grad) / ((float)graph->num_nodes * (float)graph->num_nodes));
    mean = sum_grad / (float)graph->num_nodes;

    (*coef_variation_img) = sqrt(MAX(0, variance)) / MAX(0.001, mean);

    freeMem(dist_weight);
    freeNodeAdj(&adj_rel);

    return grad;
}

//=============================================================================
// Void
//=============================================================================
inline void insertNodeInTree(Graph *graph, int index, Tree **tree, double grad)
{
    (*tree)->num_nodes++;

    for (int i = 0; i < graph->num_feats; i++)
        (*tree)->sum_feat[i] += graph->feats[index][i];

    (*tree)->sum_grad += grad;
    (*tree)->sum_grad_2 += (grad * grad);
}

inline void removeNodeFromTree(Graph *graph, int index, Tree **tree, double grad)
{
    (*tree)->num_nodes--;

    for (int i = 0; i < graph->num_feats; i++)
        (*tree)->sum_feat[i] -= graph->feats[index][i];

    (*tree)->sum_grad -= grad;
    (*tree)->sum_grad_2 -= (grad * grad);
}