package com.deep.framework.lang;

import java.util.HashMap;
import java.util.Map;

public class Index {

    static IntIndex getIndex71 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5;
    };

    static IntIndex getIndex72 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4;
    };

    static IntIndex getIndex73 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4 + index[2] * i3;
    };

    static IntIndex getIndex74 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4 + index[2] * i3 + index[3] * i2;
    };

    static IntIndex getIndex75 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4 + index[2] * i3 + index[3] * i2 + index[4] * i1;
    };

    static IntIndex getIndex76 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4 + index[2] * i3 + index[3] * i2 + index[4] * i1 + index[5] * i0;
    };

    static IntIndex getIndex77 = (int[] shape, int[] index) -> {
        int i0 = shape[6];
        int i1 = shape[5] * i0;
        int i2 = shape[4] * i1;
        int i3 = shape[3] * i2;
        int i4 = shape[2] * i3;
        int i5 = shape[1] * i4;
        return index[0] * i5 + index[1] * i4 + index[2] * i3 + index[3] * i2 + index[4] * i1 + index[5] * i0 + index[6];
    };

    static IntIndex getIndex61 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4;
    };

    static IntIndex getIndex62 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4 + index[1] * i3;
    };

    static IntIndex getIndex63 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4 + index[1] * i3 + index[2] * i2;
    };

    static IntIndex getIndex64 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4 + index[1] * i3 + index[2] * i2 + index[3] * i1;
    };

    static IntIndex getIndex65 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4 + index[1] * i3 + index[2] * i2 + index[3] * i1 + index[4] * i0;
    };

    static IntIndex getIndex66 = (int[] shape, int[] index) -> {
        int i0 = shape[5];
        int i1 = shape[4] * i0;
        int i2 = shape[3] * i1;
        int i3 = shape[2] * i2;
        int i4 = shape[1] * i3;
        return index[0] * i4 + index[1] * i3 + index[2] * i2 + index[3] * i1 + index[4] * i0 + index[5];
    };

    static IntIndex getIndex51 = (int[] shape, int[] index) -> {
        int i0 = shape[4];
        int i1 = shape[3] * i0;
        int i2 = shape[2] * i1;
        int i3 = shape[1] * i2;
        return index[0] * i3;
    };

    static IntIndex getIndex52 = (int[] shape, int[] index) -> {
        int i0 = shape[4];
        int i1 = shape[3] * i0;
        int i2 = shape[2] * i1;
        int i3 = shape[1] * i2;
        return index[0] * i3 + index[1] * i2;
    };

    static IntIndex getIndex53 = (int[] shape, int[] index) -> {
        int i0 = shape[4];
        int i1 = shape[3] * i0;
        int i2 = shape[2] * i1;
        int i3 = shape[1] * i2;
        return index[0] * i3 + index[1] * i2 + index[2] * i1;
    };

    static IntIndex getIndex54 = (int[] shape, int[] index) -> {
        int i0 = shape[4];
        int i1 = shape[3] * i0;
        int i2 = shape[2] * i1;
        int i3 = shape[1] * i2;
        return index[0] * i3 + index[1] * i2 + index[2] * i1 + index[3] * i0;
    };

    static IntIndex getIndex55 = (int[] shape, int[] index) -> {
        int i0 = shape[4];
        int i1 = shape[3] * i0;
        int i2 = shape[2] * i1;
        int i3 = shape[1] * i2;
        return index[0] * i3 + index[1] * i2 + index[2] * i1 + index[3] * i0 + index[4];
    };

    static IntIndex getIndex41 = (int[] shape, int[] index) -> {
        int i0 = shape[3];
        int i1 = shape[2] * i0;
        int i2 = shape[1] * i1;
        return index[0] * i2;
    };

    static IntIndex getIndex42 = (int[] shape, int[] index) -> {
        int i0 = shape[3];
        int i1 = shape[2] * i0;
        int i2 = shape[1] * i1;
        return index[0] * i2 + index[1] * i1;
    };

    static IntIndex getIndex43 = (int[] shape, int[] index) -> {
        int i0 = shape[3];
        int i1 = shape[2] * i0;
        int i2 = shape[1] * i1;
        return index[0] * i2 + index[1] * i1 + index[2] * i0;
    };

    static IntIndex getIndex44 = (int[] shape, int[] index) -> {
        int i0 = shape[3];
        int i1 = shape[2] * i0;
        int i2 = shape[1] * i1;
        return index[0] * i2 + index[1] * i1 + index[2] * i0 + index[3];
    };

    static IntIndex getIndex31 = (int[] shape, int[] index) -> {
        int i0 = shape[2];
        int i1 = shape[1] * i0;
        return index[0] * i1;
    };

    static IntIndex getIndex32 = (int[] shape, int[] index) -> {
        int i0 = shape[2];
        int i1 = shape[1] * i0;
        return index[0] * i1 + index[1] * i0;
    };

    public static IntIndex getIndex33 = (int[] shape, int[] index) -> {
        int i0 = shape[2];
        int i1 = shape[1] * i0;
        return index[0] * i1 + index[1] * i0 + index[2];
    };

    static IntIndex getIndex21 = (int[] shape, int[] index) -> {
        int i0 = shape[1];
        return index[0] * i0;
    };

    static IntIndex getIndex22 = (int[] shape, int[] index) -> {
        int i0 = shape[1];
        return index[0] * i0 + index[1];
    };

    static IntIndex getIndex11 = (int[] shape, int[] index) -> {
        return index[0];
    };

    static IntLength getLength1 = (int[] shape) -> {
        return shape[0];
    };

    static IntLength getLength2 = (int[] shape) -> {
        return shape[0] * shape[1];
    };

    static IntLength getLength3 = (int[] shape) -> {
        return shape[0] * shape[1] * shape[2];
    };

    static IntLength getLength4 = (int[] shape) -> {
        return shape[0] * shape[1] * shape[2] * shape[3];
    };

    static IntLength getLength5 = (int[] shape) -> {
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    };

    static IntLength getLength6 = (int[] shape) -> {
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * shape[5];
    };

    static IntLength getLength7 = (int[] shape) -> {
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4] * shape[5] * shape[6];
    };

    static Map<String, IntIndex> indexmap = new HashMap() {{
        put("getIndex71", getIndex71);
        put("getIndex72", getIndex72);
        put("getIndex73", getIndex73);
        put("getIndex74", getIndex74);
        put("getIndex75", getIndex75);
        put("getIndex76", getIndex76);
        put("getIndex77", getIndex77);

        put("getIndex61", getIndex61);
        put("getIndex62", getIndex62);
        put("getIndex63", getIndex63);
        put("getIndex64", getIndex64);
        put("getIndex65", getIndex65);
        put("getIndex66", getIndex66);

        put("getIndex51", getIndex51);
        put("getIndex52", getIndex52);
        put("getIndex53", getIndex53);
        put("getIndex54", getIndex54);
        put("getIndex55", getIndex55);

        put("getIndex41", getIndex41);
        put("getIndex42", getIndex42);
        put("getIndex43", getIndex43);
        put("getIndex44", getIndex44);

        put("getIndex31", getIndex31);
        put("getIndex32", getIndex32);
        put("getIndex33", getIndex33);

        put("getIndex21", getIndex21);
        put("getIndex22", getIndex22);

        put("getIndex11", getIndex11);
    }};

    static Map<String, IntLength> lengthMap = new HashMap() {{
        put("getLength1", getLength1);
        put("getLength2", getLength2);
        put("getLength3", getLength3);
        put("getLength4", getLength4);
        put("getLength5", getLength5);
        put("getLength6", getLength6);
        put("getLength7", getLength7);
    }};

    public static IntIndex getIndex(int[] shape, int[] index) {
        String key = "getIndex".concat(String.valueOf(shape.length)).concat(String.valueOf(index.length));
        return indexmap.get(key);
    }

    public static IntLength getIndex(int[] shape) {
        String key = "getIndex".concat(String.valueOf(shape.length));
        return lengthMap.get(key);
    }

    @FunctionalInterface
    public static interface IntIndex {
        int accept(int[] shape, int[] index);
    }

    @FunctionalInterface
    public static interface IntLength {
        int accept(int[] shape);
    }
}