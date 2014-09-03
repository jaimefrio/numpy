#include "hash_table.h"

#define MIN_HASH_SIZE 8
#define HASH_GROWTH_MUL 4
#define PERTURB_SHIFT 5

HashTable_HashFunc*
get_hash_func(PyArray_Descr *dtype)
{
    switch (dtype->type_num) {
        case NPY_BOOL:
            return &BOOL_hash_func;
            break;
        case NPY_BYTE:
            return &BYTE_hash_func;
            break;
        case NPY_UBYTE:
            return &UBYTE_hash_func;
            break;
        case NPY_SHORT:
            return &SHORT_hash_func;
            break;
        case NPY_USHORT:
            return &USHORT_hash_func;
            break;
        case NPY_INT:
            return &INT_hash_func;
            break;
        case NPY_UINT:
            return &UINT_hash_func;
            break;
        case NPY_LONG:
            return &LONG_hash_func;
            break;
        case NPY_ULONG:
            return &ULONG_hash_func;
            break;
        case NPY_LONGLONG:
            return &LONGLONG_hash_func;
            break;
        case NPY_ULONGLONG:
            return &ULONGLONG_hash_func;
            break;
        default:
            return NULL;
            break;
    }
}

HashTable*
HashTable_New(PyArrayObject *arr)
{
    size_t num_bytes = MIN_HASH_SIZE * sizeof(HashTableEntry);
    HashTable *self = malloc(sizeof(HashTable));
    if (self == NULL) {
        return NULL;
    }
    self->hash_table = NULL;
    self->dtype = PyArray_DESCR(arr);
    Py_INCREF(self->dtype);

    /* Store pointers to hashing and comparison functions */
    self->hash_func = get_hash_func(self->dtype);
    self->compare_func = self->dtype->f->compare;
    /* Allocate the hash table... */
    self->hash_table = malloc(num_bytes);
    if (self->hash_func == NULL || self->compare_func == NULL ||
                self->hash_table == NULL) {
        Py_DECREF(self->dtype);
        free(self->hash_table);
        free(self);
        return NULL;
    }
    /* ...and init all the pointers to NULL */
    memset((void *)self->hash_table, 0, num_bytes);

    /* Init the remaining attributes */
    self->mask = MIN_HASH_SIZE - 1;
    self->used = 0;

    return self;
}

HashTableEntry*
HashTable_LookUp(HashTable *self, void *value, npy_intp hash)
{
    npy_intp idx;
    npy_intp mask = self->mask;
    npy_intp perturb;
    HashTableEntry *hash_table = self->hash_table;
    HashTableEntry *hash_table_ptr;
    PyArray_CompareFunc *cmp = self->compare_func;

    idx = hash & mask;
    hash_table_ptr = hash_table + idx;
    if (hash_table_ptr->value == NULL ||
            (hash_table_ptr->hash == hash &&
             cmp(value, hash_table_ptr->value, NULL) == 0)) {
        return hash_table_ptr;
    }

    for (perturb = hash; ; perturb >>= PERTURB_SHIFT) {
        idx = 5*idx + perturb + 1;
        hash_table_ptr = hash_table + (idx & mask);
        if (hash_table_ptr->value == NULL ||
                (hash_table_ptr->hash == hash &&
                 cmp(value, hash_table_ptr->value, NULL) == 0)) {
            return hash_table_ptr;
        }

    }
}

int
HashTable_Resize(HashTable *self, int size_shift)
{
    npy_intp old_size = self->mask + 1;
    npy_intp num_items = self->used;
    npy_intp new_size = old_size << size_shift;
    npy_intp num_bytes = new_size * sizeof(HashTableEntry);
    HashTableEntry *old_table = self->hash_table;
    HashTableEntry *old_table_ptr = old_table;
    HashTableEntry *old_table_end = old_table + old_size;


    self->mask = new_size - 1;
    self->hash_table = malloc(num_bytes);
    if (self->hash_table == NULL) {
        self->hash_table = old_table;
        return -1;
    }
    memset(self->hash_table, 0, num_bytes);

    while (num_items && old_table_ptr < old_table_end) {
        // fprintf(stderr, "hash_table index: %d, num_items: %d\n", (old_table_ptr - old_table), num_items);
        void *value = old_table_ptr->value;
        if (value != NULL) {
            npy_intp hash = old_table_ptr->hash;
            HashTableEntry *new_loc = HashTable_LookUp(self, value, hash);
            new_loc->value = value;
            new_loc->hash = hash;
            num_items--;
        }
        old_table_ptr++;
    }

    return 0;
}

int
HashTable_Insert(HashTable *self, void *value)
{
    npy_intp hash = self->hash_func(value);
    HashTableEntry *hash_table_ptr = HashTable_LookUp(self, value, hash);

    if (hash_table_ptr->value == NULL) {
        hash_table_ptr->hash = hash;
        hash_table_ptr->value = value;
        self->used++;
    }

    if (3 * self->used > 2 * (self->mask + 1)) {
        return HashTable_Resize(self, self->used > 50000 ? 1 : 2);
    }

    return 0;
}

void
HashTable_CopyToArray(HashTable *self, PyArrayObject *arr)
{
    PyArray_CopySwapFunc *cpy = self->dtype->f->copyswap;
    npy_intp num_items = self->used;
    HashTableEntry *hash_table_ptr = self->hash_table;
    char *arr_data = PyArray_BYTES(arr);
    npy_intp arr_stride = PyArray_STRIDE(arr, 0);

    while (num_items) {
        void *value = hash_table_ptr->value;
        if (value != NULL) {
            cpy((void *)arr_data, value, 0, arr);
            num_items--;
            arr_data += arr_stride;
        }
        hash_table_ptr++;
    }
}

void
HashTable_Dealloc(HashTable *self)
{
    if (self != NULL) {
        Py_DECREF(self->dtype);
        free(self->hash_table);
        free(self);
    }
}