package com.deep.framework.ast;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Data
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

    public void replaceAndRemove(Node node, Node replaceNode, Node removeNode) {
        int index = getChildrens().indexOf(node);
        childrens.set(index, replaceNode);
        childrens.remove(removeNode);
    }

    public void replace(Object node, Node replaceNode) {
        int index = getChildrens().indexOf(node);
        childrens.set(index, replaceNode);
    }

    @Override
    public String toString() {
        if (childrens.isEmpty()) return "";
        String collect = childrens.stream().map(Objects::toString).collect(Collectors.joining(" "));
        if (collect.endsWith(";")) return collect;
        return collect.concat(";");
    }
}