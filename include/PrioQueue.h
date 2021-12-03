/**
* Priority Queue
*
* @date September, 2019
* @note Based on the priority heap implemented by Samuel Martins
*/
#ifndef PRIOQUEUE_H
#define PRIOQUEUE_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Structures
//=============================================================================
#include "Utils.h"

//=============================================================================
// Structures
//=============================================================================

/**
* Element removal policies
*/
typedef enum 
{
    MAXVAL_POLICY, MINVAL_POLICY
} RemPolicy;

/**
* Tie-breaking policies
*/
typedef enum
{
    LIFO_POLICY, FIFO_POLICY
} TiePolicy;

/**
* Element state within the queue. 
* WHITE_STATE: Never inserted
* GRAY_STATE: Was inserted, but never removed
* BLACK_STATE: Was inserted and orderly removed
*/
typedef enum
{
    WHITE_STATE, GRAY_STATE, BLACK_STATE
} ElemState;

/**
* Priority Queue (Heap)
*/
typedef struct 
{
    int last_elem_pos, size;
    int* pos, *node; // Node position (in heap) and node element value
    double* prio; // Priority values (clone)
    ElemState* state;
    RemPolicy rem_policy;
    TiePolicy tie_policy;
} PrioQueue;

//=============================================================================
// Bool functions
//=============================================================================
/**
* Inserts the index into the priority queue. If the priority queue is full, it
* will print a warning message, and return false.
*/
bool insertPrioQueue(PrioQueue **queue, int index);

/**
* Evaluates if the queue given in parameter is empty.
*/
bool isPrioQueueEmpty(PrioQueue *queue);

/**
* Evaluates if the queue given in parameter is full.
*/
bool isPrioQueueFull(PrioQueue *queue);

//=============================================================================
// Int functions
//=============================================================================
/**
* Computes the father's position, of a given node position
*/
int getFatherPos(int pos);

/**
* Computes the left son's position, of a given node position
*/
int getLeftSonPos(int pos);

/**
* Computes the Right son's position, of a given node position
*/
int getRightSonPos(int pos);

/**
* Removes the node top node in the heap, and returns its index. 
*/
int popPrioQueue(PrioQueue **queue);

//=============================================================================
// PrioQueue* functions
//=============================================================================
/**
* Creates an priority queue whose size is equivalent to the maximum
* index value. The priority values are defined by the array in 
* parameter, therefore, it is a sole clone of the original array.
* The removal policy is also defined in parameter, however the tie-
* breaking policy is set to FIFO as default.
*/
PrioQueue* createPrioQueue(int size, double *prio, RemPolicy rem_policy);

//=============================================================================
// Void functions
//=============================================================================
/**
* Deallocates the memory of a priority queue.
*/
void freePrioQueue(PrioQueue **queue);

/**
* Moves the index down in the priority queue. In other words, it
* reorders the heap, given the knowledge of a priority change at
* the given index.
*/
void moveIndexDownPrioQueue(PrioQueue **queue, int index);

/**
* Moves the index up in the priority queue. In other words, it
* reorders the heap, given the knowledge of a priority change at
* the given index.
*/
void moveIndexUpPrioQueue(PrioQueue **queue, int index);

/**
* Removes a specific element within the priority heap. Note
* that the state is set to WHITE_STATE, since it was not 
* ordely removed.
*/
void removePrioQueueElem(PrioQueue **queue, int index);

/**
* Resets the heap structure for the ordering of values
*/
void resetPrioQueue(PrioQueue **queue);

/**
* Sets the tie-breaking policy of the given heap
*/
void setPrioQueueTiePolicy(PrioQueue **queue, TiePolicy tie_policy);

void resetPrioQueueElem(PrioQueue **queue, int index);

#ifdef __cplusplus
}
#endif

#endif
