#ifndef __NPY_HASH_TABLE_H__
#define __NPY_HASH_TABLE_H__

#include "Python.h"
#include "numpy/arrayobject.h"

typedef npy_intp (HashTable_HashFunc)(const void *);

typedef struct {
    npy_intp hash;
    void *value;
} HashTableEntry;

typedef struct {
    npy_intp mask; /* hash_table size minus one */
    npy_intp used;
    HashTableEntry *hash_table;
    PyArray_Descr *dtype;
    HashTable_HashFunc *hash_func;
    PyArray_CompareFunc *compare_func;
} HashTable;

HashTable_HashFunc* get_hash_func(PyArray_Descr *dtype);

HashTable* HashTable_New(PyArrayObject *arr);
HashTableEntry* HashTable_LookUp(HashTable *self, void *value, npy_intp hash);
int HashTable_Insert(HashTable *self, void *value);
int HashTable_Resize(HashTable *self, int size_shift);
void HashTable_CopyToArray(HashTable *self, PyArrayObject *arr);
void HashTable_Dealloc(HashTable *self);

/*************************************/
/* HASH FUNCTIONS FOR BUILT-IN TYPES */
/*************************************/

static NPY_INLINE npy_intp
BOOL_hash_func(const void *value)
{
    return *(npy_bool *)value ? 1 : 0;
}

static NPY_INLINE npy_intp
BYTE_hash_func(const void *value)
{
    return *(npy_byte *)value;
}

static NPY_INLINE npy_intp
UBYTE_hash_func(const void *value)
{
    return *(npy_ubyte *)value;
}

static NPY_INLINE npy_intp
SHORT_hash_func(const void *value)
{
    return *(npy_short *)value;
}

static NPY_INLINE npy_intp
USHORT_hash_func(const void *value)
{
    return *(npy_ushort *)value;
}

static NPY_INLINE npy_intp
INT_hash_func(const void *value)
{
    return *(npy_int *)value;
}

static NPY_INLINE npy_intp
UINT_hash_func(const void *value)
{
    return *(npy_uint *)value;
}

static NPY_INLINE npy_intp
LONG_hash_func(const void *value)
{
    return *(npy_long *)value;
}

static NPY_INLINE npy_intp
ULONG_hash_func(const void *value)
{
    return *(npy_ulong *)value;
}

static NPY_INLINE npy_intp
LONGLONG_hash_func(const void *value)
{
    return *(npy_longlong *)value;
}

static NPY_INLINE npy_intp
ULONGLONG_hash_func(const void *value)
{
    return *(npy_ulonglong *)value;
}

#endif