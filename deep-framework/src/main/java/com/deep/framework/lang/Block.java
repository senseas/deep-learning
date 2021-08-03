package com.deep.framework.lang;

public class Block {

    public final int x, y, z;

    public Block(int x) {
        this.x = x;
        this.y = 0;
        this.z = 0;
    }

    public Block(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 0;
    }

    public Block(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

}

