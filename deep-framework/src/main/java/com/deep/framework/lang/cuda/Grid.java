package com.deep.framework.lang.cuda;

public class Grid {
    public int x, y, z;

    public Grid(int x) {
        this.x = x;
        this.y = 1;
        this.z = 1;
    }

    public Grid(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 1;
    }

    public Grid(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}

