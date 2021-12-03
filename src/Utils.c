#include "Utils.h"
#include <string.h>


//=============================================================================
// Void Functions
//=============================================================================
void freeMem(void* data)
{
    if(data != NULL) 
        free(data);
}

void printError(const char* function_name, const char* message, ...)
{
    va_list args;
    char full_msg[4096];

    va_start(args, message);
    vsprintf(full_msg, message, args);
    va_end(args);

    fprintf(stderr, "\nError in %s:\n%s!\n", function_name, full_msg);
    fflush(stdout);
    exit(0);
}

void printWarning(const char *function_name, const char *message, ...)
{
    va_list args;
    char full_msg[4096];

    va_start(args, message);
    vsprintf(full_msg, message, args);
    va_end(args);

    fprintf(stdout, "\nWarning in %s:\n%s!\n", function_name, full_msg);
}


/*
Find a string key in arguments and storage in argument variable
Input: argv and argc received in main, stringKey to match, argument variable to store match value
Output: argument variable with the match value or NULL
*/
void parseArgs(char *argv[], int argc, char *stringKey, char *argument){
    for(int i=1; i < argc-1; i++){
        // if both strings are identical
        if(strcmp(argv[i], stringKey) == 0){
            strcpy(argument, argv[i+1]);
            return;
        }
    }
    sprintf(argument, "-");
}


//=============================================================================
// Void* Functions
//=============================================================================
void* allocMem(size_t size, size_t size_bytes)
{
    void *data;

    data = calloc(size, size_bytes);

    if(data == NULL)
        printError("allocMemory", "Could not allocate memory");

    return data;
}