#include "iDISF.h"

//=============================================================================
// Auxiliary Functions
//=============================================================================
inline int addSeed(int root_index, int label, IntList *rel_seeds, int *new_labels_map, int index_label)
{
    insertIntListTail(&rel_seeds, root_index);
    new_labels_map[index_label] = label;
    index_label++;
    return index_label;
}

inline int superpixelSelectionType1(IntCell *tree_adj_head, int tree_id, int root_index, int *labels_map, int num_markers, IntList *rel_seeds, int *new_labels_map, int index_label)
{
    // se algum vizinho for objeto, não remove a semente
    bool adj_obj = false;
    for (IntCell *ptr = tree_adj_head; ptr != NULL; ptr = ptr->next)
    {
        int adj_tree_id;
        adj_tree_id = ptr->elem;

        /* adj obj or excluded seed */
        if (labels_map[adj_tree_id] <= num_markers)
        {
            adj_obj = true;
            break;
        }
    }
    if (!adj_obj)
        labels_map[tree_id] = 0;
    else // mantém a semente
        index_label = addSeed(root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);

    return index_label;
}

//=============================================================================
// Image* Functions
//=============================================================================
// iDISF with removal by class
Image *runiDISF_scribbles_rem(Graph *graph,                   // input: image graph in lab color space
                              int n_0,                        // input: desired amount of GRID seeds
                              int iterations,                 // input: maximum iterations or final superpixels (if segm_method=1)
                              Image **border_img,             // input/output: empty image to keep the superpixels borders
                              NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                              int num_markers,                // input: amount of scribbles
                              int *marker_sizes,              // input: amount of pixels in each scribble
                              int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                              int all_borders,                // input: flag. If 1, map border_img with the tree borders
                              double c1, double c2,           // input: parameters of path-cost functions 2-5
                              int obj_markers)                // input: amount of object scribbles
{
    //clock_t time;
    bool want_borders;
    int stop;
    int iter, num_cols, num_rows, num_nodes, num_feats;
    int *labels_map; // labels_map : mapeia da label 1 para a label 2;
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntList *seed_set; // passa a ser recebido na função para ser reenviado na próxima interação
    Image *label_img2;
    int *label_img;
    PrioQueue *queue;
    float size, stride, delta, max_seeds;

    stop = 0;
    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(num_nodes, sizeof(double)); // f
    //pred_map = allocMem(img->num_pixels, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    want_borders = border_img != NULL;

    grad = computeGradient(graph, &alpha);

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)n_0));
    stride = sqrtf(size) + 0.5;
    delta = stride / 2.0;
    delta = (int)delta;
    stride = (int)stride;

    // Compute the max amount of GRID seeds
    max_seeds = ((((float)num_rows - delta) / stride) + 1) * ((((float)num_cols - delta) / stride) + 1);

    // Compute the max amount of all seeds
    for (int i = 0; i < num_markers; i++)
        max_seeds += marker_sizes[i];

    labels_map = allocMem((int)max_seeds, sizeof(int)); // mapeia do indice da árvore para o rótulo da regiao

    //time = clock();
    seed_set = gridSampling_scribbles(num_rows, num_cols, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, labels_map, obj_markers);
    /*time = clock() - time;
    printf("seed sampling %.3f\n", ((double)time) / CLOCKS_PER_SEC);*/

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;

    // At least a single iteration is performed
    do
    {
        //time = clock();
        int seed_label, num_trees;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        //printf("iter %d, seeds %d \n", iter, seed_set->size);

        trees = allocMem(seed_set->size, sizeof(Tree *));
        tree_adj = allocMem(seed_set->size, sizeof(IntList *));
        are_trees_adj = allocMem(seed_set->size, sizeof(bool *));

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1; // 1 rótulo por árvore

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        seed_label = 0;
        // cria uma árvore para cada pixel de cada marcador
        // porém, cada marcador recebe um rótulo distinto
        // e as demais sementes (GRID) recebem um outro rótulo
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index;
            seed_index = ptr->elem;
            cost_map[seed_index] = 0;
            label_img[seed_index] = seed_label;
            trees[seed_label] = createTree(seed_index, num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = allocMem(seed_set->size, sizeof(bool));
            seed_label++;
            /*
	        int label = labels_map[seed_label];
            if(label > num_markers){
                NodeCoords coords = getNodeCoords(num_cols,seed_index);
            }*/
            insertPrioQueue(&queue, seed_index);
        }
        /*
        time = clock() - time;
        printf("pre IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);*/

        //time = clock();
        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree /*, mean_grad_tree*/;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            node_label2 = labels_map[node_label];
            if (node_label2 > obj_markers)
            {
                label_img2->val[node_index][0] = 2;
                node_label2 = 2;
            }
            else
            {
                label_img2->val[node_index][0] = 1;
                node_label2 = 1;
            }

            // We insert the features in the respective tree at this
            // moment, because it is guaranteed that this node will not
            // be inserted ever again.
            insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label;
                    double path_cost;

                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];
                    path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        // relação de adjacência entre árvores
                        if (node_label != adj_label)
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }

                        int adj_label2 = labels_map[adj_label];
                        if (adj_label2 > obj_markers)
                        {
                            adj_label2 = 2;
                        }
                        else
                        {
                            adj_label2 = 1;
                        }

                        // cria bordas entre rótulos (label2)
                        if (want_borders && ((all_borders == 0 && node_label2 != adj_label2) || (all_borders == 1 && node_label != adj_label)))
                        {
                            if (want_borders) // Both depicts a border between their superpixels
                            {
                                (*border_img)->val[node_index][0] = 255;
                                (*border_img)->val[adj_index][0] = 255;
                            }
                        }
                    }
                }
            }
            free(mean_feat_tree);
        }
        /*
        time = clock() - time;
        printf("IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);*/

        /*** SEED SELECTION ***/
        // Auxiliar var
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        if (stop == 2)
            stop = 1; // se já executou a uma iteracao apos remover todas as sementes grid
        if (iter < iterations && stop == 0)
        {
            //time = clock();
            //printf("Select the most relevant superpixels, iter=%d, iterations=%d \n", iter, iterations);
            // Select the most relevant superpixels
            seed_set = seedRemoval(trees, tree_adj, num_nodes, num_trees, num_markers, obj_markers, labels_map, &stop);
            /*time = clock() - time;
            printf("seed removal %.3f\n", ((double)time) / CLOCKS_PER_SEC);*/
        }

        iter++;                 // next iter
        resetPrioQueue(&queue); // Clear the queue

        /*#pragma omp parallel for firstprivate(num_trees)*/
        for (int i = 0; i < num_trees; i++)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);

        // o loop termina quando realiza a quantidade definida de iterações
    } while (iter <= iterations && stop != 1);

    freeMem(grad);
    freeMem(label_img);
    freeMem(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);
    freeMem(labels_map);

    return label_img2;
}

// iDISF with removal by relevance
Image *runiDISF(Graph *graph, // input: image graph in lab color space
                int n_0,      // input: desired amount of GRID seeds
                int n_f,
                Image **border_img,             // input/output: empty image to keep the superpixels borders
                NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                int num_user_seeds,             // input: amount of scribbles
                int *marker_sizes,              // input: amount of pixels in each scribble
                int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                int all_borders,                // input: flag. If 1, map border_img with the tree borders
                double c1, double c2,           // input: parameters of path-cost functions 2-5
                int num_objmarkers)
{
    bool want_borders;
    int num_rem_seeds, iter, num_cols, num_rows, num_nodes, num_feats;
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntList *seed_set; // passa a ser recebido na função para ser reenviado na próxima interação
    int *label_img;
    Image *label_img2;
    PrioQueue *queue;
    int scribbled_seeds = 0;

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(num_nodes, sizeof(double)); // f
    //pred_map = allocMem(num_nodes, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int));
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    want_borders = border_img != NULL;

    grad = computeGradient(graph, &alpha);
    seed_set = gridSampling(num_cols, num_rows, &n_0, coords_user_seeds, num_user_seeds, marker_sizes, grad, &scribbled_seeds, &n_f);

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;

    // At least a single iteration is performed
    do
    {
        int seed_label, num_trees, num_maintain;
        int seed_label2;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        trees = allocMem(seed_set->size, sizeof(Tree *));
        tree_adj = allocMem(seed_set->size, sizeof(IntList *));
        are_trees_adj = allocMem(seed_set->size, sizeof(bool *));

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, label_img2, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1;          // 1 rótulo por árvore
            label_img2->val[i][0] = -1; // rotula como será a segmentação de saída

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        // cria uma árvore para cada pixel de cada marcador
        // porém, o primeiro marcador (foreground) recebe um rótulo
        // e os demais marcadores e demais sementes criadas recebem outro rótulo (background)

        for (int i = 0; i < num_objmarkers; i++)
        {
            for (int j = 0; j < marker_sizes[i]; j++)
            {
                //int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
                int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
                label_img2->val[node_index][0] = 1;
            }
        }

        for (int i = num_objmarkers; i < num_user_seeds; i++)
        {
            for (int j = 0; j < marker_sizes[i]; j++)
            {
                //int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
                int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
                label_img2->val[node_index][0] = 2;
            }
        }

        seed_label = 0;
        seed_label2 = 2;

        // Assign initial values for all the seeds sampled
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        { // cria todas as árvores e as inicializa

            int seed_index;
            seed_index = ptr->elem;
            cost_map[seed_index] = 0;

            label_img[seed_index] = seed_label;
            trees[seed_label] = createTree(seed_index, num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = allocMem(seed_set->size, sizeof(bool));

            if (label_img2->val[seed_index][0] == -1)
            {
                label_img2->val[seed_index][0] = seed_label2;
            }
            else
            {
                trees[seed_label]->minDist_userSeed = 0;
            }
            if (num_user_seeds == 0)
                trees[seed_label]->minDist_userSeed = 1;

            seed_label++;
            insertPrioQueue(&queue, seed_index);
        }

        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            node_label2 = label_img2->val[node_index][0];

            // store the min distance between tree and the most closer object seed
            float min_Dist = trees[node_label]->minDist_userSeed;
            if (min_Dist > 0)
            {
                for (int j = 0; j < marker_sizes[0]; j++)
                {
                    float dist = euclDistanceCoords(node_coords, coords_user_seeds[0][j]);
                    if (dist < min_Dist)
                        trees[node_label]->minDist_userSeed = dist;
                }
            }

            insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label /*, adj_label2*/;

                    //adj_index = getNodeIndex(num_cols, adj_coords);
                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        double path_cost;
                        path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            label_img2->val[adj_index][0] = node_label2;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        if (node_label != adj_label) // If they differ, their trees are adjacent
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }
                        if (((all_borders == 0) && (node_label2 != label_img2->val[adj_index][0])) || ((all_borders == 1) && (node_label != adj_label))) // If they differ, their trees are adjacent
                        {
                            if (want_borders) // Both depicts a border between their superpixels
                            {
                                (*border_img)->val[node_index][0] = 255;
                                (*border_img)->val[adj_index][0] = 255;
                            }
                        }
                    }
                }
            }
            free(mean_feat_tree);
        }

        /*** SEED SELECTION ***/
        // Compute the number of seeds to be preserved
        num_maintain = MAX((n_0 + scribbled_seeds) * exp(-iter), scribbled_seeds + n_f);

        // Auxiliar var
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        // Select the most relevant superpixels
        seed_set = selectKMostRelevantSeeds(trees, tree_adj, num_nodes, num_trees, num_maintain, num_user_seeds);

        // Compute the number of seeds to be removed
        num_rem_seeds = num_trees - num_maintain;
        iter++;
        resetPrioQueue(&queue); // Clear the queue

        /*#pragma omp parallel for firstprivate(num_trees)*/
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

    freeMem(grad);
    freeMem(label_img);
    freeMem(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);

    return label_img2;
}

//=============================================================================
// IntList* Functions
//=============================================================================

// used in iDISF removal by class
IntList *gridSampling_scribbles(int num_rows, int num_cols, int *n_0,
                                NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, 
                                double *grad, int *labels_map, int obj_markers)
{
    float size, stride, delta_x, delta_y;
    int num_seeds, num_nodes;
    bool *label_seed;
    IntList *seed_set;
    NodeAdj *adj_rel;
    int label2;

    num_seeds = (*n_0);
    num_nodes = num_rows * num_cols;
    seed_set = createIntList();
    adj_rel = create8NeighAdj();
    label_seed = allocMem(num_nodes, sizeof(bool));

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)num_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;
    num_seeds = 0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    // mark all obj seed position true in vector
    for (int i = 0; i < num_markers; i++)
    {
        for (int j = 0; j < marker_sizes[i]; j++)
        {
            int seed_index;

            //seed_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
            seed_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;

            if (!label_seed[seed_index])
            {
                label_seed[seed_index] = true;
                labels_map[num_seeds] = i + 1;
                insertIntListTail(&seed_set, seed_index);
                num_seeds++;
            }
        }
    }
    //printf("markers seeds %d \n", seed_set->size);

    //label2 = obj_markers + 1;
    label2 = num_markers + 1;
    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if (label_seed[index])
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            j = num_cols + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    curr_coords.x = x;
                    curr_coords.y = y;

                    min_grad_index = getNodeIndex(num_cols, curr_coords);
                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        // Is valid?
                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            adj_index = getNodeIndex(num_cols, adj_coords);

                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if (label_seed[min_grad_index])
                    {
                        min_grad_index = getNodeIndex(num_cols, curr_coords);
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        labels_map[num_seeds] = label2;
                        num_seeds++;
                    }
                    else
                    {
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        labels_map[num_seeds] = label2;
                        num_seeds++;
                    }
                }
            }
        }
    }

    *n_0 = num_seeds;
    freeNodeAdj(&adj_rel);
    freeMem(label_seed);
    return seed_set;
}

// used in iDISF removal by relevance
IntList *gridSampling(int num_cols, int num_rows, int *n_0, NodeCoords **coords_user_seeds, int num_user_seeds, int *marker_sizes, double *grad, int *scribbled_seeds, int *nf)
{
    float size, stride, delta_x, delta_y;
    int num_nodes, num_seeds;
    IntList *seed_set;
    NodeAdj *adj_rel;
    bool *label_seed;

    // marcadores em linha não podem ter os pixels mudando de lugar para a posição com menor gradiente
    // porque isso pode desconectar a linha
    // a solução encontrada é só mudar os marcadores unitários (que são pontos)
    // mas é preciso verificar se a nova posição não é outra semente

    num_seeds = (*n_0);
    num_nodes = num_cols * num_rows;
    seed_set = createIntList();
    label_seed = allocMem(num_nodes, sizeof(bool));

    // Compute the approximate superpixel size and stride
    size = 0.5 + (float)(num_nodes / ((float)num_seeds + (float)num_user_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    adj_rel = create8NeighAdj();
    int count = 0;

    // change the user seed to a seed adj with min. gradient
    for (int i = 0; i < num_user_seeds; i++)
    {
        // add all point of the scribbles
        for (int j = 0; j < marker_sizes[i]; j++)
        {
            int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
            if (!label_seed[node_index])
            {
                label_seed[node_index] = true;
                insertIntListTail(&seed_set, node_index);
                (*nf)++;
                (*scribbled_seeds)++;
            }
        }
    }

    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if (label_seed[index])
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    curr_coords.x = x;
                    curr_coords.y = y;
                    min_grad_index = getNodeIndex(num_cols, curr_coords);

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            //adj_index = getNodeIndex(num_cols, adj_coords);
                            adj_index = adj_coords.y * num_cols + adj_coords.x;

                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if (label_seed[min_grad_index])
                    {
                        min_grad_index = getNodeIndex(num_cols, curr_coords);
                        //is_seed[getNodeIndex(num_cols, curr_coords)] = true;
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        count++;
                    }
                    else
                    {
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        count++;
                    }
                }
            }
        }
    }

    *n_0 = count;
    freeMem(label_seed);
    freeNodeAdj(&adj_rel);
    return seed_set;
}

// used in iDISF removal by relevance
IntList *selectKMostRelevantSeeds(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain, int num_user_seeds)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue;

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntList();
    queue = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        if (trees[i]->minDist_userSeed == 0 && num_user_seeds > 0)
        {
            // Compute the superpixel relevance
            tree_prio[i] = INFINITY;
        }
        else
        {
            double area_prio, grad_prio;
            float *mean_feat_i;

            // Compute the area relevance
            area_prio = ((float)trees[i]->num_nodes * trees[i]->minDist_userSeed) / ((float)num_nodes);

            // Initial values for the computation of gradient relevance
            grad_prio = INFINITY;
            mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

            // For each adjacent tree
            for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
            {
                int adj_tree_id;
                float *mean_feat_j;
                double dist;

                adj_tree_id = ptr->elem;
                mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);

                // Compute the L2 norm between trees
                dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

                // Get the minimum gradient value
                grad_prio = MIN(grad_prio, dist);

                free(mean_feat_j);
            }

            // Compute the superpixel relevance
            tree_prio[i] = area_prio * grad_prio;
            free(mean_feat_i);
        }
        insertPrioQueue(&queue, i);
    }

    // While it is possible to get relevant seeds
    for (int i = 0; i < num_maintain && !isPrioQueueEmpty(queue); i++)
    {
        int tree_id, root_index;

        tree_id = popPrioQueue(&queue);
        root_index = trees[tree_id]->root_index;
        insertIntListTail(&rel_seeds, root_index);
    }

    // The rest is discarted
    freePrioQueue(&queue);
    freeMem(tree_prio);
    return rel_seeds;
}

// used in iDISF removal by class
IntList *seedRemoval(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_markers, int num_objmarkers, int *new_labels_map, int *stop)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue1, *queue2, *queue3;
    double mean_relevance13, mean_relevance24; // usaremos como medida de limiar
    int *labels_map;
    int index_label;
    float sum_prio_13, sum_prio_2_13, std_deviation_13;
    float sum_prio_24, sum_prio_2_24, std_deviation_24;
    int type1 = 0, type2 = 0, type3 = 0, type4 = 0;

    index_label = 0;
    mean_relevance13 = 0.0;
    sum_prio_13 = 0;
    sum_prio_2_13 = 0;
    std_deviation_13 = 0;
    mean_relevance24 = 0.0;
    sum_prio_24 = 0;
    sum_prio_2_24 = 0;
    std_deviation_24 = 0;

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntList();
    labels_map = allocMem(num_trees, sizeof(int));

    // inicia as filas
    queue1 = createPrioQueue(num_trees, tree_prio, MINVAL_POLICY);
    queue3 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);
    queue2 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio, grad_prio_mix;
        float *mean_feat_i;
        //int root_index;
        int adjacents_type; // 0:nothing , 1:only background with background seed , 2:mixed with background seed , 3:only foreground with foreground seed, 4:mixed with foreground seed
        bool background_seed, gridSeed;
        int label;

        //root_index = trees[i]->root_index;
        label = new_labels_map[i];
        labels_map[i] = label;
        background_seed = false;
        gridSeed = false;

        adjacents_type = 0;

        // Compute the area relevance
        area_prio = (double)trees[i]->num_nodes / (double)num_nodes;

        // Initial values for the computation of gradient relevance
        grad_prio = INFINITY;
        grad_prio_mix = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

        if (label > num_objmarkers)
            background_seed = true;
        if (label > num_markers)
            gridSeed = true;

        // For each adjacent tree
        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id, adj_label;
            float *mean_feat_j;
            double dist;
            bool adj_background;

            adj_background = false;
            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);
            adj_label = new_labels_map[adj_tree_id];

            // Compute the L2 norm between trees
            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            if (adj_label > num_objmarkers) // labels_map : indice indica labels e valor indica labels2
                adj_background = true;

            // Get the minimum gradient value
            if ((background_seed && !adj_background) || (!background_seed && adj_background))
                grad_prio_mix = MIN(grad_prio_mix, dist);
            else
                grad_prio = MIN(grad_prio, dist);

            if (adjacents_type != 2 && adjacents_type != 4)
            {
                /* os tipos com seed background são 1 e 2. */
                if (background_seed)
                {
                    if (!adj_background)
                        adjacents_type = 2;
                    else
                        adjacents_type = 1;
                }
                /* os tipos com seed foreground são 3 e 4. */
                else
                {
                    if (adj_background)
                        adjacents_type = 4;
                    else
                        adjacents_type = 3;
                }
            }
            free(mean_feat_j);
        }

        // Compute the superpixel relevance
        switch (adjacents_type)
        {
        case 1:
            tree_prio[i] = area_prio * grad_prio;
            type1++;
            if (gridSeed)
                insertPrioQueue(&queue1, i);
            else
                index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 2:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type2++;
            if (gridSeed)
                insertPrioQueue(&queue2, i);
            else
                index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;

        case 3:
            tree_prio[i] = area_prio * grad_prio;
            type3++;
            //insertPrioQueue(&queue3, i);
            index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 4:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type4++;
            index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;
        }
        free(mean_feat_i);
    }

    std_deviation_13 = sqrtf(fabs(sum_prio_2_13 - 2 * sum_prio_13 + ((sum_prio_13 * sum_prio_13) / (float)type1 + type3)) / (float)type1 + type3);
    mean_relevance13 = sum_prio_13 / (float)(type1 + type3);
    mean_relevance13 += std_deviation_13;

    std_deviation_24 = sqrtf(fabs(sum_prio_2_24 - 2 * sum_prio_24 + ((sum_prio_24 * sum_prio_24) / (float)type2 + type4)) / (float)type2 + type4);
    mean_relevance24 = sum_prio_24 / (float)(type2 + type4);
    mean_relevance24 += std_deviation_24;

    while (!isPrioQueueEmpty(queue1))
    {
        int tree_id;

        tree_id = popPrioQueue(&queue1);
        if (tree_prio[tree_id] >= mean_relevance13)
        {
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
            while (!isPrioQueueEmpty(queue1))
            {
                tree_id = popPrioQueue(&queue1);
                index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
            }
        }
        else
        {
            index_label = superpixelSelectionType1(tree_adj[tree_id]->head, tree_id, trees[tree_id]->root_index, labels_map, num_objmarkers, rel_seeds, new_labels_map, index_label);
        }
    }

    int prio = mean_relevance13;
    while (!isPrioQueueEmpty(queue3) && prio >= mean_relevance13)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue3);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance13)
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
    }

    prio = mean_relevance24;
    while (!isPrioQueueEmpty(queue2) && prio >= mean_relevance24)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance24)
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
    }

    bool background = false, foreground = false, gridSeeds = false;
    if (num_markers == 1)
        background = true;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers)
            foreground = true;
        else
        {
            if (new_labels_map[i] > num_markers)
                gridSeeds = true;
            else
                background = true;
        }
        if (foreground && background && gridSeeds)
            break;
    }
    if (num_markers == 1)
        background = false;
    if (!gridSeeds)
        *stop = 2;
    if (!foreground || (!background && !gridSeeds))
        *stop = 1;

    /*
    bool background = false, foreground = false;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers) foreground = true;
        else background = true;
        if (foreground && background) break;
    }
    if (!foreground || !background) *stop = 1;
    */

    //printf("stop=%d \n", *stop);

    freePrioQueue(&queue1);
    freePrioQueue(&queue3);
    freePrioQueue(&queue2);
    freeMem(tree_prio);
    freeMem(labels_map);
    return rel_seeds;
}
