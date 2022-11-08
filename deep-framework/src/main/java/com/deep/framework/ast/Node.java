package com.deep.framework.ast;

import java.util.ArrayList;
import java.util.List;

public class Node {
    private Node prarent;
    private List<Object> childrens;

    public Node() {
        this.childrens = new ArrayList<>();
    }

    public Node(Node prarent) {
        this.prarent = prarent;
        this.childrens = new ArrayList<>();
    }

    public Node getPrarent() {
        return prarent;
    }

    public List<Object> getChildrens() {
        return childrens;
    }

    public void setChildrens(List<Object> childrens) {
        this.childrens = childrens;
    }
}
