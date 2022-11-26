package com.deep.framework.ast;

import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;

import java.util.Objects;
import java.util.stream.Collectors;

public class Node extends Token {
    private Node prarent;
    private NodeList<Node> childrens;

    public Node() {
        super(null);
        this.childrens = new NodeList<>();
    }

    public Node(TokenType type) {
        super(type);
        this.childrens = new NodeList<>();
    }

    public Node(Node prarent) {
        super(null);
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
        getChildrens().remove(removeNode);
    }

    public void replace(Node node, Node replaceNode) {
        int index = childrens.indexOf(node);
        childrens.set(index, replaceNode);
    }

    public void setPrarent(Node prarent) {
        this.prarent = prarent;
    }

    @Override
    public String toString() {
        if (Objects.nonNull(getTokenType())) return getTokenType().toString();
        if (childrens.isEmpty()) return "";
        return childrens.stream().map(Objects::toString).collect(Collectors.joining(" "));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        return getTokenType() == o;
    }
}