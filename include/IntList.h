/**
* Single-Linked Integer List
*
* @date September, 2019
*/
#ifndef INTLIST_H
#define INTLIST_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Utils.h"

//=============================================================================
// Structures
//=============================================================================

/**
* Abstract Integer Cell
*/
typedef struct IntCell
{
    int elem;
    struct IntCell* next;
} IntCell;

/**
* Single-linked Integer List
*/
typedef struct
{  
    int size;
    IntCell* head;
    IntCell* tail;
} IntList;

//=============================================================================
// Bool Functions
//=============================================================================
/**
* Evaluates if the list is empty.
*/
bool isIntListEmpty(IntList *list);

/**
* Evaluates if an specific element exists in the list.
*/
bool existsIntListElem(IntList *list, int elem);

/**
* Inserts an element at the given index of the list.
*/
bool insertIntListAt(IntList **list, int elem, int index);

/**
* Inserts an element as the head of the list.
*/
bool insertIntListHead(IntList **list, int elem);

/**
* Inserts an element at the end of the list.
*/
bool insertIntListTail(IntList **list, int elem);

//=============================================================================
// Int Functions
//=============================================================================

/**
* Removes the element at the given index. If the list is empty, it prints 
* a warning message.
*/
int removeIntListAt(IntList **list, int index);

/**
* Removes the list's head. If the list is empty, it prints a warning message.
*/
int removeIntListHead(IntList **list);

/**
* Removes the list's tail. If the list is empty, it prints a warning message.
*/
int removeIntListTail(IntList **list);

//=============================================================================
// IntList* Functions
//=============================================================================
/**
* Creates an empty list of integers.
*/
IntList *createIntList();

//=============================================================================
// IntCell* Functions
//=============================================================================
/**
* Creates an integer cell containing the given element.
*/
IntCell *createIntCell(int elem);

//=============================================================================
// Void Functions
//=============================================================================
/**
* Deallocates the memory reserved for the list given in parameter
*/
void freeIntList(IntList **list);

/**
* Deallocates the memory reserved for the integer cell given in parameter
*/
void freeIntCell(IntCell **node);

/**
* Removes the desired element in the list if it exists. If not, it prints a
* warning message.
*/
void removeIntListElem(IntList **list, int elem);

void clearIntList(IntList **list);

/**
* Prints the elements within the given list
*/
void printIntList(IntList *list);

#ifdef __cplusplus
}
#endif

#endif
