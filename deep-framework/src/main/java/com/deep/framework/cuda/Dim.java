package com.deep.framework.cuda;

public class Dim {
    public int x, y, z;

    public Dim(int x) {
        this.x = x;
        this.y = 1;
        this.z = 1;
    }

    public Dim(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 1;
    }

    public Dim(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}