package com.deep.framework.lang.cuda;

public class Block {
    public int x, y, z;

    public Block(int x) {
        this.x = x;
        this.y = 1;
        this.z = 1;
    }

    public Block(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 1;
    }

    public Block(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}