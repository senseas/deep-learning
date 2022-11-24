package com.deep.framework.ast;

import com.deep.framework.ast.lexer.TokenType;

import java.util.Objects;
import java.util.stream.Collectors;

public class Node {
    private Node prarent;
    private NodeList<Node> childrens;

    public Node() {
        this.childrens = new NodeList<>();
    }

    public Node(Node prarent) {
        this.prarent = prarent;
        this.childrens = new NodeList<>();
    }

    public Node getPrarent() {
        return prarent;
    }

    public NodeList<Node> getChildrens() {
        return childrens;
    }

    public void setChildrens(NodeList<Node> childrens) {
        this.childrens = childrens;
    }

    public void replaceAndRemove(Node node, Node replaceNode, Object removeNode) {
        replace(node, replaceNode);
        remove(removeNode);
    }

    public Node get(TokenType tokenType) {
        return childrens.stream().filter(a -> a.equals(tokenType)).findFirst().get();
    }

    public void replace(Node node, Node replaceNode) {
        int index = childrens.indexOf(node);
        childrens.set(index, replaceNode);
    }

    public void remove(Object removeNode) {
        getChildrens().remove(removeNode);
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