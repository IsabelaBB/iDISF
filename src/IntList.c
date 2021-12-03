#include "IntList.h"

//=============================================================================
// Constructors & Deconstructors
//=============================================================================
IntCell* createIntCell(int elem)
{
    IntCell *node;

    node = (IntCell*)calloc(1, sizeof(IntCell));

    node->elem = elem;
    node->next = NULL;

    return node;
}

IntList* createIntList()
{
    IntList* list;

    list = (IntList*)calloc(1, sizeof(IntList));

    list->size = 0;
    list->head = NULL;
    list->tail = NULL;

    return list;
}

void freeIntCell(IntCell** node)
{
    if(*node != NULL)
    {
        IntCell* tmp;

        tmp = *node;

        tmp->next = NULL;
        
        free(tmp);

        *node = NULL;
    }
}

void freeIntList(IntList **list)
{
    if(*list != NULL)
    {
        IntCell *tmp;

        (*list)->tail = NULL;

        tmp = (*list)->head;

        while(tmp != NULL)
        {
            IntCell *prev;

            prev = tmp;
            tmp = tmp->next;

            freeIntCell(&prev);
        }

        free(tmp);
        free(*list);
        *list = NULL;
    }
}


//=============================================================================
// Bool Functions
//=============================================================================
inline bool isIntListEmpty(IntList *list)
{
    return list->size == 0;
}

bool existsIntListElem(IntList *list, int elem)
{    
    bool exists;

    if(isIntListEmpty(list))
    {
        exists = false;
    }
    else
    {
        IntCell *cell;

        cell = list->head;

        while(cell != NULL && cell->elem != elem);

        exists = cell == NULL;
    }

    return exists;
}

bool insertIntListAt_bkp(IntList **list, int elem, int index)
{
    if(!isIntListEmpty(*list))
        index = 0; // Force to put in the list head
    
    bool success;
    int i;
    IntCell *prev_cell, *curr_cell, *new_cell;
    IntList *tmp;

    i = 0;
    tmp = *list;
    prev_cell = NULL;
    curr_cell = tmp->head;

    while(i != index)
    {
        prev_cell = curr_cell;
        curr_cell = curr_cell->next;
        i++;
    }

    new_cell = createIntCell(elem);

    if(prev_cell != NULL){
        prev_cell->next = new_cell;
        printf("%d -> %d , index = %d \n", prev_cell->elem, new_cell->elem, index);
    }
    else{
        tmp->head = new_cell;
        printf("%d , index = %d \n", new_cell->elem, index);
    }

    new_cell->next = curr_cell;

    tmp->size++;

    success = true;

    return success;
}

bool insertIntListAt(IntList **list, int elem, int index)
{
    if(isIntListEmpty(*list))
        index = 0; // Force to put in the list head

    bool success;
    int i;
    IntCell *prev_cell, *curr_cell, *new_cell;
    IntList *tmp;

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

    new_cell = createIntCell(elem);

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

inline bool insertIntListHead(IntList **list, int elem)
{    
    return insertIntListAt(list, elem, 0);
}

inline bool insertIntListTail(IntList **list, int elem)
{
    if(isIntListEmpty(*list)){
        return insertIntListAt(list, elem, 0);
    }
    else{
        return insertIntListAt(list, elem, (*list)->size);   
    }
}


//=============================================================================
// Int Functions
//=============================================================================

int removeIntListAt(IntList **list, int index)
{
    int elem_rem;

    elem_rem = -1;
    if(isIntListEmpty(*list))
       printWarning("removeIntListAt", "List of seeds is empty");
    else if(index < 0 || index >= (*list)->size)
        printError("removeIntListAt", "Index is out of bounds: %d", index);
    else
    {
        int i;
        IntCell *rem_cell, *prev_cell, *next_cell;
        IntList *tmp;

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
        freeIntCell(&rem_cell);

        tmp->size--;
    }

    return elem_rem;
}

inline int removeIntListHead(IntList **list)
{
    return removeIntListAt(list, 0);
}

inline int removeIntListTail(IntList **list)
{
    return removeIntListAt(list, (*list)->size - 1);
}

//=============================================================================
// Void Functions
//=============================================================================

void clearIntList(IntList **list){
    if(*list != NULL)
    {
        IntCell *tmp;

        (*list)->tail = NULL;

        tmp = (*list)->head;

        while(tmp != NULL)
        {
            IntCell *prev;

            prev = tmp;
            tmp = tmp->next;

            freeIntCell(&prev);
        }

        free(tmp);
        (*list)->head = NULL;
        //free(*list);
        //*list = NULL;
    }
}

void removeIntListElem(IntList **list, int elem)
{
    if(isIntListEmpty(*list))
        printWarning("removeIntListElem", "List of seeds is empty");
    else 
    {
        IntCell *rem_cell, *prev_cell, *next_cell;
        IntList *tmp;   

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
        
            freeIntCell(&rem_cell);
            tmp->size--;
        }
        else
            printWarning("removeIntListElem", "The element to be removed was not found");
    }
}