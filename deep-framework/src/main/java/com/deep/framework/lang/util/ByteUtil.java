package com.deep.framework.lang.util;

import java.nio.*;

public class ByteUtil {

    /**
     * {@link double[]} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(double[] input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.length * (Double.SIZE / 8));
        for (double a : input) {
            buffer.putDouble(a);
        }
        return buffer;
    }

    /**
     * {@link float[]} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(float[] input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.length * (Float.SIZE / 8));
        for (float a : input) {
            buffer.putFloat(a);
        }
        return buffer;
    }

    /**
     * {@link float[]} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(long[] input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.length * (Long.SIZE / 8));
        for (long a : input) {
            buffer.putLong(a);
        }
        return buffer;
    }

    /**
     * {@link int[]} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(int[] input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.length * (Integer.SIZE / 8));
        for (int a : input) {
            buffer.putInt(a);
        }
        return buffer;
    }

    /**
     * {@link int[]} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(short[] input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.length * (Short.SIZE / 8));
        for (short a : input) {
            buffer.putShort(a);
        }
        return buffer;
    }

    /**
     * {@link DoubleBuffer} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(DoubleBuffer input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.capacity() * (Double.SIZE / 8));
        while (input.hasRemaining()) {
            buffer.putDouble(input.get());
        }
        return buffer;
    }

    /**
     * {@link FloatBuffer} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(FloatBuffer input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.capacity() * (Float.SIZE / 8));
        while (input.hasRemaining()) {
            buffer.putFloat(input.get());
        }
        return buffer;
    }

    /**
     * {@link LongBuffer} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(LongBuffer input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.capacity() * (Long.SIZE / 8));
        while (input.hasRemaining()) {
            buffer.putLong(input.get());
        }
        return buffer;
    }

    /**
     * {@link IntBuffer} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(IntBuffer input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.capacity() * (Integer.SIZE / 8));
        while (input.hasRemaining()) {
            buffer.putInt(input.get());
        }
        return buffer;
    }

    /**
     * {@link ShortBuffer} TO {@link ByteBuffer}
     * @param input
     * @return
     */
    public static ByteBuffer asByteBuffer(ShortBuffer input) {
        if (null == input) {
            return null;
        }
        ByteBuffer buffer = ByteBuffer.allocate(input.capacity() * (Short.SIZE / 8));
        while (input.hasRemaining()) {
            buffer.putShort(input.get());
        }
        return buffer;
    }

    /**
     * {@link double[]} TO {@link byte[]}
     * @param input
     * @return byte[]
     */
    public static byte[] asByteArray(double[] input) {
        if (null == input) {
            return null;
        }
        return asByteBuffer(DoubleBuffer.wrap(input)).array();
    }

    /**
     * {@link float[]} TO {@link byte[]}
     * @param input
     * @return byte[]
     */
    public static byte[] asByteArray(float[] input) {
        if (null == input) {
            return null;
        }
        return asByteBuffer(FloatBuffer.wrap(input)).array();
    }

    /**
     * {@link long[]} TO {@link byte[]}
     * @param input
     * @return byte[]
     */
    public static byte[] asByteArray(long[] input) {
        if (null == input) {
            return null;
        }
        return asByteBuffer(LongBuffer.wrap(input)).array();
    }

    /**
     * {@link int[]} TO {@link byte[]}
     * @param input
     * @return byte[]
     */
    public static byte[] asByteArray(int[] input) {
        if (null == input) {
            return null;
        }
        return asByteBuffer(IntBuffer.wrap(input)).array();
    }

    /**
     * {@link short[]} TO {@link byte[]}
     * @param input
     * @return byte[]
     */
    public static byte[] asByteArray(short[] input) {
        if (null == input) {
            return null;
        }
        return asByteBuffer(ShortBuffer.wrap(input)).array();
    }

    /**
     * {@link byte[]} TO {@link double[]}
     * @param input
     * @return double[]
     */
    public static double[] asDoubleArray(byte[] input) {
        if (null == input) {
            return null;
        }
        DoubleBuffer buffer = ByteBuffer.wrap(input).asDoubleBuffer();
        double[] res = new double[buffer.remaining()];
        buffer.get(res);
        return res;
    }

    /**
     * {@link byte[]} TO {@link float[]}
     * @param input
     * @return float[]
     */
    public static float[] asFloatArray(byte[] input) {
        if (null == input) {
            return null;
        }
        FloatBuffer buffer = ByteBuffer.wrap(input).asFloatBuffer();
        float[] res = new float[buffer.remaining()];
        buffer.get(res);
        return res;
    }

    /**
     * {@link byte[]} TO {@link long[]}
     * @param input
     * @return
     */
    public static long[] asLongArray(byte[] input) {
        if (null == input) {
            return null;
        }
        LongBuffer buffer = ByteBuffer.wrap(input).asLongBuffer();
        long[] res = new long[buffer.remaining()];
        buffer.get(res);
        return res;
    }

    /**
     * {@link byte[]} TO {@link int[]}
     * @param input
     * @return int[]
     */
    public static int[] asIntArray(byte[] input) {
        if (null == input) {
            return null;
        }
        IntBuffer buffer = ByteBuffer.wrap(input).asIntBuffer();
        int[] res = new int[buffer.remaining()];
        buffer.get(res);
        return res;
    }

    /**
     * {@link byte[]} TO {@link short[]}
     * @param input
     * @return short[]
     */
    public static short[] asShortArray(byte[] input) {
        if (null == input) {
            return null;
        }
        ShortBuffer buffer = ByteBuffer.wrap(input).asShortBuffer();
        short[] res = new short[buffer.remaining()];
        buffer.get(res);
        return res;
    }

}