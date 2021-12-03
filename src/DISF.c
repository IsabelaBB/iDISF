#include "DISF.h"

//=============================================================================
// Image* Functions
//=============================================================================

Image *runLabeledDISF(Graph *graph, int n_0, int n_f, NodeCoords **coords_user_seeds, Image **border_img)
{
    bool want_borders;
    int num_rem_seeds, iter;
    double *cost_map;
    NodeAdj *adj_rel;
    IntList *seed_set;
    Image *label_img;
    PrioQueue *queue;

    // Aux
    cost_map = (double *)calloc(graph->num_nodes, sizeof(double));
    // adj_rel = create4NeighAdj();
    adj_rel = create8NeighAdj();
    label_img = createImage(graph->num_rows, graph->num_cols, 1);
    queue = createPrioQueue(graph->num_nodes, cost_map, MINVAL_POLICY);

    want_borders = border_img != NULL;

    seed_set = gridSamplingDISF(graph, n_0);

    iter = 1; // At least a single iteration is performed
    do
    {
        int seed_label, num_trees, num_maintain;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        trees = (Tree **)calloc(seed_set->size, sizeof(Tree *));
        tree_adj = (IntList **)calloc(seed_set->size, sizeof(IntList *));
        are_trees_adj = (bool **)calloc(seed_set->size, sizeof(bool *));

// Initialize values
#pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            label_img->val[i][0] = -1;

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        seed_label = 0;
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index;

            seed_index = ptr->elem;

            cost_map[seed_index] = 0;
            label_img->val[seed_index][0] = seed_label;

            trees[seed_label] = createTree(seed_index, graph->num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = (bool *)calloc(seed_set->size, sizeof(bool));

            seed_label++;
            insertPrioQueue(&queue, seed_index);
        }

        // IFT algorithm
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label;
            NodeCoords node_coords;
            float *mean_feat_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(graph->num_cols, node_index);
            node_label = label_img->val[node_index][0];

            // This node won't appear here ever again
            insertNodeInTree(graph, node_index, &(trees[node_label]), 1);

            mean_feat_tree = meanTreeFeatVector(trees[node_label]);

            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
                {
                    int adj_index, adj_label;

                    adj_index = getNodeIndex(graph->num_cols, adj_coords);
                    adj_label = label_img->val[adj_index][0];

                    // If it wasn't inserted nor orderly removed from the queue
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        double arc_cost, path_cost;

                        arc_cost = euclDistance(mean_feat_tree, graph->feats[adj_index], graph->num_feats);

                        path_cost = MAX(cost_map[node_index], arc_cost);

                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img->val[adj_index][0] = node_label;

                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else if (node_label != adj_label) // Their trees are adjacent
                    {
                        if (want_borders) // Both depicts a border between their superpixels
                        {
                            (*border_img)->val[node_index][0] = 255;
                            (*border_img)->val[adj_index][0] = 255;
                        }

                        if (!are_trees_adj[node_label][adj_label])
                        {
                            insertIntListTail(&(tree_adj[node_label]), adj_label);
                            insertIntListTail(&(tree_adj[adj_label]), node_label);
                            are_trees_adj[adj_label][node_label] = true;
                            are_trees_adj[node_label][adj_label] = true;
                        }
                    }
                }
            }

            free(mean_feat_tree);
        }

        num_maintain = MAX(n_0 * exp(-iter), n_f);

        // Aux
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        seed_set = selectSeedDISF(trees, tree_adj, graph->num_nodes, num_trees, num_maintain);
        num_rem_seeds = num_trees - seed_set->size;

        iter++;
        resetPrioQueue(&queue);

        for (int i = 0; i < num_trees; ++i)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);
    } while (num_rem_seeds > 0);

    int obj_index = getNodeIndex(graph->num_cols, coords_user_seeds[0][0]);
    int tree_id_obj = label_img->val[obj_index][0];

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (label_img->val[i][0] == tree_id_obj)
        {
            label_img->val[i][0] = 1;
        }
        else
            label_img->val[i][0] = 2;
        (*border_img)->val[i][0] = 0;
    }

    for (int i = 0; i < graph->num_nodes; i++)
    {
        int node_label = label_img->val[i][0];
        NodeCoords node_coords = getNodeCoords(graph->num_cols, i);

        for (int j = 0; j < adj_rel->size; j++)
        {
            NodeCoords adj_coords;
            adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, j);

            if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
            {
                int adj_index, adj_label;
                adj_index = getNodeIndex(graph->num_cols, adj_coords);
                adj_label = label_img->val[adj_index][0];

                if (node_label != adj_label && want_borders)
                {
                    (*border_img)->val[i][0] = 255;
                    (*border_img)->val[adj_index][0] = 255;
                }
            }
        }
    }

    free(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);

    return label_img;
}


//=============================================================================
// IntList* Functions
//=============================================================================

// used in DISF
IntList *gridSamplingDISF(Graph *graph, int num_seeds)
{
    float size, stride, delta_x, delta_y;
    double *grad;
    bool *is_seed;
    IntList *seed_set;
    NodeAdj *adj_rel;

    seed_set = createIntList();
    is_seed = (bool *)calloc(graph->num_nodes, sizeof(bool));

    // Approximate superpixel size
    size = 0.5 + (float)(graph->num_nodes / (float)num_seeds);
    stride = sqrtf(size) + 0.5;

    delta_x = delta_y = stride / 2.0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    double variation;
    grad = computeGradient(graph, &variation);
    adj_rel = create8NeighAdj();

    for (int y = (int)delta_y; y < graph->num_rows; y += stride)
    {
        for (int x = (int)delta_x; x < graph->num_cols; x += stride)
        {
            int min_grad_index;
            NodeCoords curr_coords;

            curr_coords.x = x;
            curr_coords.y = y;

            min_grad_index = getNodeIndex(graph->num_cols, curr_coords);

            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
                {
                    int adj_index;

                    adj_index = getNodeIndex(graph->num_cols, adj_coords);

                    if (grad[adj_index] < grad[min_grad_index])
                        min_grad_index = adj_index;
                }
            }

            is_seed[min_grad_index] = true;
        }
    }

    for (int i = 0; i < graph->num_nodes; i++)
        if (is_seed[i]) // Assuring unique values
            insertIntListTail(&seed_set, i);

    free(grad);
    free(is_seed);
    freeNodeAdj(&adj_rel);

    return seed_set;
}


IntList *selectSeedDISF(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue;

    tree_prio = (double *)calloc(num_trees, sizeof(double));
    rel_seeds = createIntList();
    queue = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio;
        float *mean_feat_i;

        area_prio = trees[i]->num_nodes / (float)num_nodes;

        grad_prio = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]);

        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id;
            float *mean_feat_j;
            double dist;

            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);

            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            grad_prio = MIN(grad_prio, dist);

            free(mean_feat_j);
        }

        tree_prio[i] = area_prio * grad_prio;

        insertPrioQueue(&queue, i);

        free(mean_feat_i);
    }

    for (int i = 0; i < num_maintain && !isPrioQueueEmpty(queue); i++)
    {
        int tree_id, root_index;

        tree_id = popPrioQueue(&queue);
        root_index = trees[tree_id]->root_index;

        insertIntListTail(&rel_seeds, root_index);
    }

    freePrioQueue(&queue); // The remaining are discarded
    free(tree_prio);

    return rel_seeds;
}


