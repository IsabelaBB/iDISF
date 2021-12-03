#include "Image.h"

//=============================================================================
// Constructors & Deconstructors
//=============================================================================
Image *createImage(int num_rows, int num_cols, int num_channels)
{
    Image *new_img;
    int i;

    new_img = (Image*)calloc(1, sizeof(Image));

    new_img->num_rows = num_rows;
    new_img->num_cols = num_cols;
    new_img->num_pixels = num_rows * num_cols;
    new_img->num_channels = num_channels;

    new_img->val = (int**)calloc(new_img->num_pixels, sizeof(int*));
/*#pragma omp parallel for \
    private(i) \
    firstprivate(num_channels, num_rows, num_cols) \
    shared(new_img)*/
    for(i = 0; i < num_rows * num_cols; i++)
        new_img->val[i] = (int*)calloc(num_channels, sizeof(int));

    return new_img;
}

void freeImage(Image **img)
{
    int num_pixels;

    if(*img != NULL)
    {
        Image *tmp;
        tmp = *img;
        int i;
        num_pixels = tmp->num_pixels;

    #pragma omp parallel for private(i) \
        firstprivate(num_pixels) \
        shared(tmp)
        for(i = 0; i < num_pixels; i++)
            freeMem(tmp->val[i]);
        freeMem(tmp->val);

        freeMem(tmp);

        *img = NULL;
    }
}

//=============================================================================
// Int Functions
//=============================================================================
int getMaximumValue(Image *img, int channel)
{
    int max_val, chn_begin, chn_end;
    int num_pixels, i,j;

    num_pixels = img->num_pixels;
    max_val = -1;
    
    if(channel == -1)
    {
        chn_begin = 0; chn_end = img->num_channels - 1;
    }
    else chn_begin = chn_end = channel;

    /*#pragma omp parallel for \
        private(i,j) \
        firstprivate(num_pixels, chn_begin, chn_end) \
        shared(img) \
        reduction(max:max_val)*/
    for(i = 0; i < num_pixels; i++)
        for(j = chn_begin; j <= chn_end; j++)
            if(max_val < img->val[i][j])
                max_val = img->val[i][j];

    return max_val;   
}

int getMinimumValue(Image *img, int channel)
{
    int min_val, chn_begin, chn_end, num_pixels, i,j;

    num_pixels = img->num_pixels;
    
    if(channel == -1)
    {
        chn_begin = 0; chn_end = img->num_channels - 1;
    }
    else chn_begin = chn_end = channel;

    min_val = img->val[0][chn_begin];

    /*#pragma omp parallel for \
        private(i,j) \
        firstprivate(num_pixels, chn_begin, chn_end) \
        shared(img) \
        reduction(min:min_val)*/
    for(i = 0; i < num_pixels; i++)
        for(j = chn_begin; j <= chn_end; j++)
            if(min_val > img->val[i][j])
                min_val = img->val[i][j];

    return min_val;   
}

int getNormValue(Image *img)
{
    int max_val;

    max_val = getMaximumValue(img, -1);

    if(max_val > 65535)
        printError("getNormValue", "This code supports only 8-bit and 16-bit images!");

    if(max_val <= 255) return 255;
    else return 65535;
}

/*
//=============================================================================
// Image* Functions
//=============================================================================
Image *loadImage(const char* filepath)
{
    int num_channels, num_rows, num_cols, num_pixels, i,j;
    unsigned char *data;    
    Image *new_img;
    
    data = stbi_load(filepath, &num_cols, &num_rows, &num_channels, 0);

    if(data == NULL)
        printError("loadImage", "Could not load the image <%s>", filepath);

    new_img = createImage(num_rows, num_cols, num_channels);
    num_channels = new_img->num_channels;
    num_pixels = new_img->num_pixels;

#pragma omp parallel for \
    private(i,j) \
    firstprivate(num_channels, num_pixels) \
    shared(new_img, data)
    for(i = 0; i < num_pixels; i++)
    {
        new_img->val[i] = (int*)calloc(num_channels, sizeof(int));

        for(j = 0; j < num_channels; j++)
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

#pragma omp parallel for \
    private(i,j) \
    firstprivate(num_channels,num_pixels,img_num_channels,normval,BORDER_sRGB) \
    shared(ovlay_img,border_img, img)
    for(i = 0; i < num_pixels; i++)
    {
        for(j = 0; j < num_channels; j++)
        {
            if(border_img->val[i][0] != 0)
                ovlay_img->val[i][j] = BORDER_sRGB[j] * normval;
            
            else if(img_num_channels == 1) // It will convert the image to PPM
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
void writeImagePPM(Image *img, char* filepath)
{
    int max_val, min_val;
    FILE *fp;

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fp = fopen(filepath, "wb");

    if(fp == NULL)
        printError("writeImagePPM", "Could not open the file %s", filepath);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PPM file
    if(max_val < 256 && min_val >= 0)
    {
        unsigned char* rgb;

        rgb = allocMem(img->num_channels, sizeof(unsigned char));

        for(int i = 0; i < img->num_pixels; i++)
        {
            for(int c = 0; c < img->num_channels; c++)
                rgb[c] = img->val[i][c];

            fwrite(rgb, 1, img->num_channels, fp);
        }

        freeMem(rgb);
    }
    // 16-bit PPM file
    else if(max_val < 65536 && min_val >= 0)
    {
        unsigned short* rgb;

        rgb = allocMem(img->num_channels, sizeof(unsigned short));

        for(int i = 0; i < img->num_pixels; i++)
        {
            for(int c = 0; c < img->num_channels; c++)
                rgb[c] = ((img->val[i][c] & 0xff) << 8) | ((unsigned short)img->val[i][c] >> 8);

            fwrite(rgb, 2, img->num_channels, fp);
        }

        freeMem(rgb); 
    }
    else
        printError("writeImagePPM", "Invalid max and/or min vals %d, %d", max_val, min_val);

    fclose(fp);
}

void writeImagePGM(Image *img, char* filepath)
{
    int max_val, min_val;
    FILE *fp;

    fp = fopen(filepath, "wb");

    if(fp == NULL)
        printError("writeImagePGM", "Could not open the file <%s>", filepath);

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PGM file
    if(max_val < 256 && min_val >= 0)
    {
        unsigned char* data;

        data = (unsigned char*)calloc(img->num_pixels, sizeof(unsigned char));

        for(int i = 0; i < img->num_pixels; i++)
            data[i] = (unsigned char)img->val[i][0];

        fwrite(data, sizeof(unsigned char), img->num_pixels, fp);

        free(data);
    }
    // 16-bit PGM file
    else if(max_val < 65536 && min_val >= 0) 
    {
        unsigned short* data;

        data = (unsigned short*)calloc(img->num_pixels, sizeof(unsigned short));

        for(int i = 0; i < img->num_pixels; i++)
            data[i] = (unsigned short)img->val[i][0];
        
        for(int i = 0; i < img->num_pixels; i++)
        {
            int high, low;

            high = ((data[i]) & 0x0000FF00) >> 8;
            low = (data[i]) & 0x000000FF;

            fputc(high,fp);
            fputc(low, fp);
        }

        free(data);   
    }
    else
        printError("writeImagePGM", "Invalid min/max spel values <%d,%d>", min_val, max_val);

    fclose(fp);
}
*/