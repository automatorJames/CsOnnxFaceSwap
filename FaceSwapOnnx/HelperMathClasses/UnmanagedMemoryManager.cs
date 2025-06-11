namespace FaceSwapOnnx.HelperMathClasses;

using System;
using System.Buffers;
using System.Runtime.InteropServices;

public unsafe class UnmanagedMemoryManager<T> : MemoryManager<T> where T : unmanaged
{
    private T* _pointer;
    private int _length;

    public UnmanagedMemoryManager(T* pointer, int length)
    {
        _pointer = pointer;
        _length = length;
    }

    public override Span<T> GetSpan()
    {
        return new Span<T>(_pointer, _length);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex >= _length)
            throw new ArgumentOutOfRangeException(nameof(elementIndex));
        return new MemoryHandle(_pointer + elementIndex);
    }

    public override void Unpin()
    {
        // No action needed since memory is already pinned
    }

    protected override void Dispose(bool disposing)
    {
        // No action needed since we don't own the memory
    }
}

