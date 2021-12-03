/**
* Single-Linked Integer List
*
* @date September, 2019
*/
#ifndef INTLABELEDLIST_H
#define INTLABELEDLIST_H

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
typedef struct IntLabeledCell
{
    int elem;
    int label;
    int treeId;
    bool recompute;
    struct IntLabeledCell* next;
} IntLabeledCell;

/**
* Single-linked Integer List
*/
typedef struct
{  
    int size;
    IntLabeledCell* head;
    IntLabeledCell* tail;
} IntLabeledList;

//=============================================================================
// Bool Functions
//=============================================================================
/**
* Evaluates if the list is empty.
*/
bool isIntLabeledListEmpty(IntLabeledList *list);

/**
* Evaluates if an specific element exists in the list.
*/
bool existsIntLabeledListElem(IntLabeledList *list, int elem, int *label, int *treeId);

/**
* Inserts an element at the given index of the list.
*/
bool insertIntLabeledListAt(IntLabeledList **list, int elem, int index, int label, int treeId, bool recompute);

/**
* Inserts an element as the head of the list.
*/
bool insertIntLabeledListHead(IntLabeledList **list, int elem, int label, int treeId);

/**
* Inserts an element at the end of the list.
*/
bool insertIntLabeledListTail(IntLabeledList **list, int elem, int label, int treeId);

bool insertIntLabeledListTail_rec(IntLabeledList **list, int elem, int label, int treeId, bool recompute);

//=============================================================================
// Int Functions
//=============================================================================

/**
* Removes the element at the given index. If the list is empty, it prints 
* a warning message.
*/
int removeIntLabeledListAt(IntLabeledList **list, int index);

/**
* Removes the list's head. If the list is empty, it prints a warning message.
*/
int removeIntLabeledListHead(IntLabeledList **list);

/**
* Removes the list's tail. If the list is empty, it prints a warning message.
*/
int removeIntLabeledListTail(IntLabeledList **list);

//=============================================================================
// IntLabeledList* Functions
//=============================================================================
/**
* Creates an empty list of integers.
*/
IntLabeledList *createIntLabeledList();

//=============================================================================
// IntCell* Functions
//=============================================================================
/**
* Creates an integer cell containing the given element.
*/
IntLabeledCell *createIntLabeledCell(int elem, int label, int treeId, bool recompute);

//=============================================================================
// Void Functions
//=============================================================================
/**
* Deallocates the memory reserved for the list given in parameter
*/
void freeIntLabeledList(IntLabeledList **list);

/**
* Deallocates the memory reserved for the integer cell given in parameter
*/
void freeIntLabeledCell(IntLabeledCell **node);

/**
* Removes the desired element in the list if it exists. If not, it prints a
* warning message.
*/
void removeIntLabeledListElem(IntLabeledList **list, int elem);

/**
* Prints the elements within the given list
*/
void printIntLabeledList(IntLabeledList *list);

#ifdef __cplusplus
}
#endif

#endif
