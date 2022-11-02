/**
* Interactive Segmentation based on Dynamic and Iterative Spanning Forest (C)
*
* @date August, 2021
*/

//=============================================================================
// Includes
//=============================================================================
#include "Image.h"
#include "iDISF.h"
#include "Utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//=============================================================================
// Prototypes
//=============================================================================
void usage(char *argv);
Image *loadImage(const char *filepath);
void writeImagePGM(Image *img, char *filepath);
void writeImagePPM(Image *img, char *filepath);
Image *overlayBorders(Image *img, Image *border_img);


void usage_iDISF_scribbles_rem(char *argv)
{
    printf("\n Usage    : %s --rem 1 [options]\n", argv); // 5 args
    printf("\n Options  : \n");
    printf("           --i : Input image\n");
    printf("           --n0 : Number of init GRID seeds (>= 0)\n");
    printf("           --it : Iterations \n");
    printf("           --f : Path-cost function. { 1:color distance, 2:gradient-cost, 3:beta norm, 4:cv tree norm, 5:sum gradient-cost, 6: sum beta norm }\n");
    printf("\n Optional : \n");
    printf("           --xseeds : Object seed x coord (for a single seed)\n");
    printf("           --yseeds : Object seed y coord (for a single seed)\n");
    printf("           --file <scribbles.txt>: File name with the pixel coordinates of the scribbles\n");
    printf("           WARNING: Use --xseeds --yseeds OR --file\n");
    printf("           --inverse : Inverse the pixel coordinates of the scribbles. Can use any char with this flag to activate it. \n");
    printf("           --c1 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --c2 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --max_markers : Define the number of scribbles that will be used. (Default: The number of scribbles equal to the scribbles file)\n");
    printf("           --obj_markers : Define the number of scribbles that will be labeled as object. (Default: all scribbles are labeled as object)\n");
    printf("           --o <path/image>: Define the output image name and its path.\n");
    printf("\n Seeds file format : \n");
    printf("          number of scribbles\\n \n");
    printf("          number of pixels in the scribble\\n \n");
    printf("          x_coord;y_coord\\n \n");
    printf("          obs: the last coord don't have \"\\n\" \n");
}

void usage_iDISF(char *argv)
{
    printf("\n Usage    : %s --rem 3 [options]\n", argv); // 5 args
    printf("\n Options  : \n");
    printf("           --i : Input image\n");
    printf("           --n0 : Number of init GRID seeds (>= 0)\n");
    printf("           --nf : Number of final superpixels\n");
    printf("           --f : Path-cost function. {1:color distance, 2:gradient-cost, 3:beta norm, 4:cv tree norm, 5:sum gradient-cost, 6: sum beta norm}\n");
    printf("\n Optional : \n");
    printf("           --xseeds : Object seed x coord\n");
    printf("           --yseeds : Object seed y coord\n");
    printf("           --file <scribbles.txt>: File name with the pixel coordinates of the scribbles\n");
    printf("           WARNING: Use --xseeds --yseeds OR --file\n");
    printf("           --inverse : Inverse the pixel coordinates of the scribbles. Can use any char with this flag to activate it. \n");
    printf("           --c1 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --c2 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --max_markers : Define the number of scribbles that will be used. (Default: The number of scribbles in scribbles file)\n");
    printf("           --obj_markers : Define the number of scribbles that will be labeled as object. (Default: all scribbles are labeled as object)\n");
    printf("           --o <path/image>: Define the image name and its path.\n");
    printf("\n Seeds file format : \n");
    printf("          number of scribbles\\n \n");
    printf("          number of pixels in the scribble\\n \n");
    printf("          x_coord;y_coord\\n \n");
    printf("          obs: the last coord don't have \"\\n\" \n");
}


/**
 * Reads the coordinates of the marked pixels in a text file 
 * @param fileMarkers in : char[255] -- a file name
 * @param max_markers in : int -- the maximum number of markers
 * @param marker_sizes out : &int[num_user_seeds] -- the size of each marker
 * @param coords_markers out : &NodeCoords[num_user_seeds][marker_sizes]
 * @param inverse in : 0 or 1 -- indicates if the coordinates are inverted
 * @return : number of markers 
 */
int readMarkersFile(char fileMarkers[],
                    int max_markers,
                    int **marker_sizes_out,
                    NodeCoords ***coords_markers_out,
                    int inverse)
{
    int num_user_seeds = 0;
    NodeCoords **coords_markers;
    int *marker_sizes;

    FILE *file = fopen(fileMarkers, "r");

    if (file == NULL)
        printError("readFile", ("Was do not possible to read the seeds file"));

    if (fscanf(file, "%d\n", &num_user_seeds) < 1)
        printError("readFile", "Invalid file format");

    marker_sizes = (int *)allocMem(num_user_seeds, sizeof(int));
    coords_markers = (NodeCoords **)allocMem(num_user_seeds, sizeof(NodeCoords *));

    if (max_markers < 1)
        max_markers = num_user_seeds;

    for (int i = 0; i < MIN(num_user_seeds, max_markers); i++)
    {
        if (fscanf(file, "%d\n", &(marker_sizes[i])) < 1)
            printError("readFile", "Invalid file format");

        coords_markers[i] = (NodeCoords *)allocMem(marker_sizes[i], sizeof(NodeCoords));

        for (int j = 0; j < marker_sizes[i]; j++)
        {
            if (!(i == num_user_seeds - 1 && j == marker_sizes[i] - 1))
            {
                if (inverse == 0)
                {
                    if (fscanf(file, "%d;%d\n", &coords_markers[i][j].x, &coords_markers[i][j].y) != 2 ||
                        coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                    {
                        printf("coordsx=%d, coordsy=%d \n", coords_markers[i][j].x, coords_markers[i][j].y);
                        printError("readFile", "Invalid coords values in file");
                    }
                }
                else
                {
                    // CASO AS COORDENADAS ESTEJAM TROCADAS
                    if (fscanf(file, "%d;%d\n", &coords_markers[i][j].y, &coords_markers[i][j].x) != 2 ||
                        coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                        printError("readFile", "Invalid coords values in file");
                }
            }
        }
    }

    int i = num_user_seeds - 1;
    int j = marker_sizes[i] - 1;

    if (num_user_seeds <= max_markers)
    {
        if (inverse == 0)
        {
            if (fscanf(file, "%d;%d", &coords_markers[i][j].x, &coords_markers[i][j].y) != 2 ||
                coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                printError("readFile", "Invalid coords values in file2");
        }
        else
        {
            // CASO AS COORDENADAS ESTEJAM TROCADAS
            if (fscanf(file, "%d;%d", &coords_markers[i][j].y, &coords_markers[i][j].x) != 2 ||
                coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                printError("readFile", "Invalid coords values in file2");
        }
    }
    else
    {
        num_user_seeds = max_markers;
    }

    fclose(file);

    (*coords_markers_out) = coords_markers;
    (*marker_sizes_out) = marker_sizes;

    return num_user_seeds;
}



//=============================================================================
// Main
//=============================================================================

// iDISF with removal by class
Image *main_iDISF_scribbles_rem(char *argv, Graph *graph, int n_0, int iterations, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, int obj_markers)
{
    Image *label_img;

    // Validation of user's inputs
    if (n_0 < 0 || iterations <= 0 || function <= 0 || function > 6 || num_markers < 1)
    {
        usage_iDISF_scribbles_rem(argv);
        printError("main", "Too many/few parameters or invalid params values!");
    }

    label_img = runiDISF_scribbles_rem(graph, n_0, iterations, border_img, coords_user_seeds, num_markers, marker_sizes, function, all_borders, c1, c2, obj_markers);
    return label_img;
}


// iDISF with removal by relevance
Image *main_iDISF(char *argv, Graph *graph, int n_0, int n_f, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, int obj_markers)
{
    Image *label_img;

    // Validation of user's inputs
    if (n_0 < 0 || n_f < 1 || function <= 0 || function > 6 || num_markers < 1)
    {
        usage_iDISF(argv);
        printError("main", "Too many/few parameters or invalid params values!");
    }

    label_img = runiDISF(graph, n_0, n_f, border_img, coords_user_seeds, num_markers, marker_sizes, function, all_borders, c1, c2, obj_markers);
    return label_img;
}


int main(int argc, char *argv[])
{
    // input args
    char imagePath[255];
    char fileObjSeeds[255];
    char fileSeeds[256];
    int n_0, iterations, n_f;
    int function;
    int all_borders;
    double c1, c2;
    int max_markers;
    int obj_markers;
    int seedRemoveOption;
    int i;

    // structures used
    Image *img, *border_img, *label_img, *ovlay_img;
    NodeCoords **coords_user_seeds;
    Graph *graph;
    clock_t time;

    // others
    int num_markers, *marker_sizes = NULL;
    char *pch, *imageName;
    char ovlayName[255], labelsName[255], bordersName[255];
    char xseeds[255], yseeds[255], n0[255], it[255], f[255];
    char inverse[255];
    char all[255];
    char c1_char[256], c2_char[256];
    char output[256];
    char maxMarkers[256];
    char objMarkers[256];
    char fileName[256];
    char nf[256];
    char seedRemoveOption_char[256];

    // get arguments
    parseArgs(argv, argc, (char *)"--rem", seedRemoveOption_char);
    parseArgs(argv, argc, (char *)"--i", imagePath);
    parseArgs(argv, argc, (char *)"--n0", n0);
    parseArgs(argv, argc, (char *)"--f", f);
    
    parseArgs(argv, argc, (char *)"--nf", nf); // used only in iDISF with removal by relevance
    parseArgs(argv, argc, (char *)"--it", it); // used only in iDISF with removal by class

    parseArgs(argv, argc, (char *)"--xseeds", xseeds);
    parseArgs(argv, argc, (char *)"--yseeds", yseeds);
    parseArgs(argv, argc, (char *)"--file", fileObjSeeds);
    
    parseArgs(argv, argc, (char *)"--inverse", inverse);
    parseArgs(argv, argc, (char *)"--all", all);
    parseArgs(argv, argc, (char *)"--c1", c1_char);
    parseArgs(argv, argc, (char *)"--c2", c2_char);
    parseArgs(argv, argc, (char *)"--max_markers", maxMarkers);
    parseArgs(argv, argc, (char *)"--obj_markers", objMarkers);
    parseArgs(argv, argc, (char *)"--o", output);
    

    // mandatory args
    iterations = atoi(it);
    function = atoi(f);

    if (strcmp(seedRemoveOption_char, "-") != 0)
    {
        int tmp = atoi(seedRemoveOption_char);
        if (tmp > 0 && tmp <= 4)
            seedRemoveOption = tmp;
        else
            seedRemoveOption = 1;
    }
    else
        seedRemoveOption = 1;

    // some optional args
    if (strcmp(n0, "-") != 0)
        n_0 = atoi(n0);
    else
        n_0 = -1; // dont reduces the object seed set

    if (strcmp(nf, "-") != 0)
        n_f = atoi(nf);
    else
        n_f = -1;

    if (strcmp(all, "-") != 0)
        all_borders = 1;
    else
        all_borders = 0;

    if (strcmp(maxMarkers, "-") != 0)
        max_markers = atoi(maxMarkers);
    else
        max_markers = -1;

    if (strcmp(c1_char, "-") != 0)
        c1 = atof(c1_char);
    else
        c1 = 0.7;

    if (strcmp(c2_char, "-") != 0)
        c2 = atof(c2_char);
    else
        c2 = 0.8;


    // Load image and get the user-defined params
    if (strcmp(imagePath, "-") == 0)
    {
        printError("main", "ERROR: Missing image path with flag --i!");
    }
    img = loadImage(imagePath);

    // Get object seeds coords in a txt file
    if (strcmp(fileObjSeeds, "-") != 0)
        num_markers = readMarkersFile(fileObjSeeds, max_markers, &marker_sizes, &coords_user_seeds, strcmp(inverse, "-"));
    else
    {
        // get one point object seed
        if (strcmp(xseeds, "-") != 0 && strcmp(yseeds, "-") != 0)
        {
            num_markers = 1;
            marker_sizes = (int *)allocMem(num_markers, sizeof(int));
            marker_sizes[0] = 1;
            coords_user_seeds = (NodeCoords **)allocMem(num_markers, sizeof(NodeCoords *));
            coords_user_seeds[0] = (NodeCoords *)allocMem(1, sizeof(NodeCoords));
            coords_user_seeds[0][0].x = atoi(xseeds);
            coords_user_seeds[0][0].y = atoi(yseeds);
        }
        else
        {
            // dont have object seed
            num_markers = 0;
            coords_user_seeds = NULL;
        }
    }

    if (strcmp(objMarkers, "-") != 0)
        obj_markers = MIN(num_markers, MAX(1, atoi(objMarkers)));
    else
        obj_markers = num_markers;

    // Get only the image name
    pch = strtok(imagePath, "/");
    imageName = pch;
    while (pch != NULL)
    {
        imageName = pch;
        pch = strtok(NULL, "/");
    }
    pch = strtok(imageName, ".");
    imageName = pch;

    // Create auxiliary data structures
    border_img = createImage(img->num_rows, img->num_cols, 1);
    graph = createGraph(img);

    //printf("Segmentation mode \n");
        time = clock();
        switch (seedRemoveOption)
        {
        case 1:
            //printf("running runiDISF_scribbles_rem \n");
            label_img = main_iDISF_scribbles_rem(argv[0], graph, n_0, iterations, &border_img, coords_user_seeds, num_markers, marker_sizes, function, all_borders, c1, c2, obj_markers);
            break;

        case 2:
            //printf("running runiDISF \n");
            label_img = main_iDISF(argv[0], graph, n_0, n_f, &border_img, coords_user_seeds, num_markers, marker_sizes, function, all_borders, c1, c2, obj_markers);
            break;

        default:
            break;
        }
        time = clock() - time;

        freeGraph(&graph);

        // Overlay superpixel's borders into a copy of the original image
        ovlay_img = overlayBorders(img, border_img);
        freeImage(&img);

        printf("%.3f\n\n", ((double)time) / CLOCKS_PER_SEC);

        // read the image name to write in results
        if (strcmp(output, "-") != 0)
        {
            sprintf(ovlayName, "%s_ovlay.ppm", output);
            sprintf(labelsName, "%s_labels.pgm", output);
            sprintf(bordersName, "%s_borders.pgm", output);
        }
        else
        {
            switch (seedRemoveOption)
            {
            case 1:
                sprintf(ovlayName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_ovlay.ppm", imageName, n_0, iterations, c1, c2, function, num_markers, seedRemoveOption);
                sprintf(labelsName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_labels.pgm", imageName, n_0, iterations, c1, c2, function, num_markers, seedRemoveOption);
                sprintf(bordersName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_borders.pgm", imageName, n_0, iterations, c1, c2, function, num_markers, seedRemoveOption);
                break;
            case 2:
                sprintf(ovlayName, "%s_n0-%d_nf-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_ovlay.ppm", imageName, n_0, n_f, c1, c2, function, num_markers, seedRemoveOption);
                sprintf(labelsName, "%s_n0-%d_nf-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_labels.pgm", imageName, n_0, n_f, c1, c2, function, num_markers, seedRemoveOption);
                sprintf(bordersName, "%s_n0-%d_nf-%d_c1-%.1f_c2-%.1f_f%d_numMarkers%d_segm%d_borders.pgm", imageName, n_0, n_f, c1, c2, function, num_markers, seedRemoveOption);
                break;
            
            default:
                break;
            }
        }

        // Save the segmentation results
        writeImagePPM(ovlay_img, ovlayName);
        writeImagePGM(label_img, labelsName);
        writeImagePGM(border_img, bordersName);

        // Free
        for (i = 0; i < num_markers; ++i)
        {
            freeMem(coords_user_seeds[i]);
        }
        freeImage(&img);
        freeImage(&label_img);
        freeImage(&border_img);
        freeImage(&ovlay_img);
        freeMem(coords_user_seeds);
        free(marker_sizes);
    
}

//=============================================================================
// Image* Functions
//=============================================================================

Image *loadImage(const char *filepath)
{
    int num_channels, num_rows, num_cols, num_nodes;
    int i, j;
    unsigned char *data;
    Image *new_img;

    data = stbi_load(filepath, &num_cols, &num_rows, &num_channels, 0);

    if (data == NULL)
        printError("loadImage", "Could not load the image <%s>", filepath);

    new_img = createImage(num_rows, num_cols, num_channels);
    num_nodes = num_rows * num_cols;

    /*#pragma omp parallel for private(i, j)    \
    firstprivate(num_nodes, num_channels) \
        shared(new_img, data)*/
    for (i = 0; i < num_nodes; i++)
    {
        new_img->val[i] = (int *)calloc(num_channels, sizeof(int));

        for (j = 0; j < num_channels; j++)
        {
            new_img->val[i][j] = data[i * num_channels + j];
        }
    }

    stbi_image_free(data);

    return new_img;
}


Image *overlayBorders(Image *img, Image *border_img)
{
    const float BORDER_sRGB[] = {0.0, 1.0, 1.0}; // Cyan

    int normval, i, j, num_channels, img_num_channels, num_pixels;
    Image *ovlay_img;

    normval = getNormValue(img);
    ovlay_img = createImage(img->num_rows, img->num_cols, 3);
    num_channels = ovlay_img->num_channels;
    img_num_channels = img->num_channels;
    num_pixels = img->num_pixels;

#pragma omp parallel for private(i, j)                                \
    firstprivate(num_channels, num_pixels, img_num_channels, normval) \
        shared(ovlay_img, border_img, img, BORDER_sRGB)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < num_channels; j++)
        {
            if (border_img->val[i][0] != 0)
                ovlay_img->val[i][j] = BORDER_sRGB[j] * normval;

            else if (img_num_channels == 1) // It will convert the image to PPM
                ovlay_img->val[i][j] = img->val[i][0];
            else
                ovlay_img->val[i][j] = img->val[i][j];
        }
    }

    return ovlay_img;
}

//=============================================================================
// Void Functions
//=============================================================================
void writeImagePPM(Image *img, char *filepath)
{

    int max_val, min_val;
    FILE *fp;

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fp = fopen(filepath, "wb");

    if (fp == NULL)
        printError("writeImagePPM", "Could not open the file %s", filepath);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PPM file
    if (max_val < 256 && min_val >= 0)
    {
        unsigned char *rgb;

        rgb = (unsigned char *)allocMem(img->num_channels, sizeof(unsigned char));

        for (int i = 0; i < img->num_pixels; i++)
        {
            for (int c = 0; c < img->num_channels; c++)
                rgb[c] = img->val[i][c];

            fwrite(rgb, 1, img->num_channels, fp);
        }

        freeMem(rgb);
    }
    // 16-bit PPM file
    else if (max_val < 65536 && min_val >= 0)
    {
        unsigned short *rgb;

        rgb = (unsigned short *)allocMem(img->num_channels, sizeof(unsigned short));

        for (int i = 0; i < img->num_pixels; i++)
        {
            for (int c = 0; c < img->num_channels; c++)
                rgb[c] = ((img->val[i][c] & 0xff) << 8) | ((unsigned short)img->val[i][c] >> 8);

            fwrite(rgb, 2, img->num_channels, fp);
        }

        freeMem(rgb);
    }
    else
        printError("writeImagePPM", "Invalid max and/or min vals %d, %d", max_val, min_val);

    fclose(fp);
}

void writeImagePGM(Image *img, char *filepath)
{
    int max_val, min_val, i;
    FILE *fp;
    int num_pixels;

    num_pixels = img->num_pixels;

    fp = fopen(filepath, "wb");

    if (fp == NULL)
        printError("writeImagePGM", "Could not open the file <%s>", filepath);

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);
    //printf("max_val:%d, min_val:%d \n", max_val, min_val);

    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PGM file
    if (max_val < 256 && min_val >= 0)
    {
        unsigned char *data;

        data = (unsigned char *)calloc(num_pixels, sizeof(unsigned char));

#pragma omp parallel for private(i) \
    firstprivate(num_pixels)        \
        shared(data, img)
        for (i = 0; i < num_pixels; i++)
            data[i] = (unsigned char)img->val[i][0];

        fwrite(data, sizeof(unsigned char), num_pixels, fp);

        free(data);
    }
    // 16-bit PGM file
    else if (max_val < 65536 && min_val >= 0)
    {
        unsigned short *data;

        data = (unsigned short *)calloc(num_pixels, sizeof(unsigned short));

#pragma omp parallel for private(i) \
    firstprivate(num_pixels)        \
        shared(data, img)
        for (i = 0; i < num_pixels; i++)
            data[i] = (unsigned short)img->val[i][0];

        for (i = 0; i < num_pixels; i++)
        {
            int high, low;

            high = ((data[i]) & 0x0000FF00) >> 8;
            low = (data[i]) & 0x000000FF;

            fputc(high, fp);
            fputc(low, fp);
        }

        free(data);
    }
    else
        printError("writeImagePGM", "Invalid min/max spel values <%d,%d>", min_val, max_val);

    fclose(fp);
}
