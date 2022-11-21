package com.deep.framework.ast;

import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

@Data
public class Node {
    private Node prarent;
    private List<Node> childrens;

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

    public List<Node> getChildrens() {
        return childrens;
    }

    public Object getLastChild() {
        return childrens.get(childrens.size() - 1);
    }

    public void setChildrens(List<Node> childrens) {
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

    public boolean contains(TokenType tokenType) {
        return childrens.stream().anyMatch(a -> a.equals(tokenType));
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