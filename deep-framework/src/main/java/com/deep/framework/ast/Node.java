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

    public Object getLastChild() {
        return childrens.get(childrens.size() - 1);
    }

    public void setChildrens(List<Object> childrens) {
        this.childrens = childrens;
    }

    public void replaceAndRemove(Node node, Node replaceNode, Object removeNode) {
        replace(node, replaceNode);
        remove(removeNode);
    }

    public void replaceAndRemove(Node node, Node replaceNode, Object removeNode, Object removeNodeb) {
        replace(node, replaceNode);
        remove(removeNode);
        remove(removeNodeb);
    }

    public void replace(Object node, Node replaceNode) {
        for (int index = 0; index < getChildrens().size(); index++) {
            if (getChildrens().get(index) == node) {
                childrens.set(index, replaceNode);
            }
        }
    }

    public void remove(Object removeNode) {
        for (int index = 0; index < getChildrens().size(); index++) {
            if (getChildrens().get(index) == removeNode) {
                getChildrens().remove(index);
            }
        }
    }

    public Node setPrarent(Node prarent) {
        this.prarent = prarent;
        return this;
    }

    @Override
    public String toString() {
        if (childrens.isEmpty()) return "";
        String collect = childrens.stream().map(Objects::toString).collect(Collectors.joining(" "));
        if (collect.endsWith(";")) return collect;
        return collect.concat(";");
    }
}