#include "IntLabeledList.h"

//=============================================================================
// Constructors & Deconstructors
//=============================================================================
IntLabeledCell* createIntLabeledCell(int elem, int label, int treeId, bool recompute)
{
    IntLabeledCell *node;

    node = (IntLabeledCell*)calloc(1, sizeof(IntLabeledCell));

    node->elem = elem;
    node->label = label;
    node->treeId = treeId;
    node->next = NULL;
    node->recompute = recompute;

    return node;
}

IntLabeledList* createIntLabeledList()
{
    IntLabeledList* list;

    list = (IntLabeledList*)calloc(1, sizeof(IntLabeledList));

    list->size = 0;
    list->head = NULL;
    list->tail = NULL;
    return list;
}

void freeIntLabeledCell(IntLabeledCell **node)
{
    if(*node != NULL)
    {
        IntLabeledCell* tmp;
        tmp = *node;
        tmp->next = NULL;
        free(tmp);
        *node = NULL;
    }
}

void freeIntLabeledList(IntLabeledList **list)
{
    if(*list != NULL)
    {
        IntLabeledCell *tmp;

        (*list)->tail = NULL;

        tmp = (*list)->head;

        while(tmp != NULL)
        {
            IntLabeledCell *prev;

            prev = tmp;
            tmp = tmp->next;

            freeIntLabeledCell(&prev);
        }

        free(tmp);
        free(*list);
        *list = NULL;
    }
}


//=============================================================================
// Bool Functions
//=============================================================================
inline bool isIntLabeledListEmpty(IntLabeledList *list)
{
    return list->size == 0;
}

bool existsIntLabeledListElem(IntLabeledList *list, int elem, int *label, int *treeId)
{    
    bool exists;
    (*label) = -1;
    (*treeId) = -1;

    if(isIntLabeledListEmpty(list))
    {
        exists = false;
    }
    else
    {
        IntLabeledCell *cell;

        cell = list->head;

        while(cell != NULL && cell->elem != elem);

        exists = cell == NULL;
        if(exists){
            (*label) = cell->label;
            (*treeId) = cell->treeId;
        }
    }
    return exists;
}

bool insertIntLabeledListAt(IntLabeledList **list, int elem, int index, int label, int treeId, bool recompute)
{
    if(isIntLabeledListEmpty(*list))
        index = 0; // Force to put in the list head

    bool success;
    int i;
    IntLabeledCell *prev_cell, *curr_cell, *new_cell;
    IntLabeledList *tmp;

    i = 0;
    tmp = *list;
    prev_cell = NULL;
    
    // insert a node in tail of list
    if(tmp->size == index && tmp->size > 0)
    {
        prev_cell = tmp->tail;
        curr_cell = NULL;
    } 
    else // insert a node in head or middle of list
    {
        curr_cell = tmp->head;
        while(i != index)
        {
            prev_cell = curr_cell;
            curr_cell = curr_cell->next;
            i++;
        }
    }

    new_cell = createIntLabeledCell(elem, label, treeId, recompute);

    if(prev_cell != NULL){
        prev_cell->next = new_cell;
    }
    else{
        tmp->head = new_cell;
    }

    if(index == tmp->size){
        tmp->tail = new_cell;
    }

    new_cell->next = curr_cell;
    tmp->size++;
    success = true;

    return success;
}

inline bool insertIntLabeledListHead(IntLabeledList **list, int elem, int label, int treeId)
{    
    return insertIntLabeledListAt(list, elem, 0, label, treeId, false);
}

inline bool insertIntLabeledListTail(IntLabeledList **list, int elem, int label, int treeId)
{
    if(isIntLabeledListEmpty(*list)){
        return insertIntLabeledListAt(list, elem, 0, label, treeId, false);
    }
    else{
        return insertIntLabeledListAt(list, elem, (*list)->size, label, treeId, false);   
    }
}

inline bool insertIntLabeledListTail_rec(IntLabeledList **list, int elem, int label, int treeId, bool recompute)
{
    if(isIntLabeledListEmpty(*list)){
        return insertIntLabeledListAt(list, elem, 0, label, treeId, recompute);
    }
    else{
        return insertIntLabeledListAt(list, elem, (*list)->size, label, treeId, recompute);   
    }
}

//=============================================================================
// Int Functions
//=============================================================================

int removeIntLabeledListAt(IntLabeledList **list, int index)
{
    int elem_rem;

    elem_rem = -1;
    if(isIntLabeledListEmpty(*list))
       printWarning("removeIntLabeledListAt", "List of seeds is empty");
    else if(index < 0 || index >= (*list)->size)
        printError("removeIntLabeledListAt", "Index is out of bounds: %d", index);
    else
    {
        int i;
        IntLabeledCell *rem_cell, *prev_cell, *next_cell;
        IntLabeledList *tmp;

        tmp = *list;
        i = 0;
        prev_cell = NULL;
        rem_cell = tmp->head;
        next_cell = rem_cell->next;

        while(i != index)
        {
            prev_cell = rem_cell;
            rem_cell = rem_cell->next;
            next_cell = rem_cell->next;
            i++;
        }

        if(prev_cell == NULL)
            tmp->head = next_cell;
        else
            prev_cell->next = next_cell;

        if(tmp->size == index+1)
            tmp->tail = prev_cell;
        
        elem_rem = rem_cell->elem;
        freeIntLabeledCell(&rem_cell);

        tmp->size--;
    }

    return elem_rem;
}

inline int removeIntLabeledListHead(IntLabeledList **list)
{
    return removeIntLabeledListAt(list, 0);
}

inline int removeIntLabeledListTail(IntLabeledList **list)
{
    return removeIntLabeledListAt(list, (*list)->size - 1);
}

//=============================================================================
// Void Functions
//=============================================================================

void removeIntLabeledListElem(IntLabeledList **list, int elem)
{
    if(isIntLabeledListEmpty(*list))
        printWarning("removeIntLabeledListElem", "List of seeds is empty");
    else 
    {
        IntLabeledCell *rem_cell, *prev_cell, *next_cell;
        IntLabeledList *tmp;   

        tmp = *list;

        prev_cell = next_cell = NULL;
        rem_cell = tmp->head;
        next_cell = rem_cell->next;

        while(rem_cell != NULL && rem_cell->elem != elem)
        {
            prev_cell = rem_cell;
            rem_cell = rem_cell->next;
            next_cell = rem_cell->next;
        }

        if(rem_cell != NULL)
        {
            if(prev_cell == NULL)
                tmp->head = next_cell;
            else
                prev_cell->next = next_cell;

            if(rem_cell->elem == tmp->tail->elem)
                tmp->tail = prev_cell;
        
            freeIntLabeledCell(&rem_cell);
            tmp->size--;
        }
        else
            printWarning("removeIntLabeledListElem", "The element to be removed was not found");
    }
}