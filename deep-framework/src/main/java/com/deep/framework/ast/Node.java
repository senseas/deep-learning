package com.deep.framework.ast;

import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.BlockStatement;

import java.lang.reflect.Type;
import java.util.Objects;
import java.util.stream.Collectors;

public class Node extends Token {
    private Node prarent;
    private NodeList<Node> childrens;

    public Node() {
        super(null);
    }

    public Node(TokenType type) {
        super(type);
    }

    public Node(Node prarent) {
        super(null);
        this.prarent = prarent;
    }

    public Node getPrarent() {
        return prarent;
    }

    public NodeList<Node> getChildrens() {
        if (Objects.nonNull(childrens)) return childrens;
        return childrens = new NodeList<>();
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

    public boolean isName() {
        return this instanceof Name;
    }

    public boolean isBlockStatement() {
        return this instanceof BlockStatement;
    }

    public boolean endsTypeof(Type clas) {
        return childrens.get(childrens.size() - 1).getClass().equals(clas);
    }

    public boolean typeof(Type clas) {
        return this.getClass().equals(clas);
    }

    public boolean notTypeof(Type clas) {
        return !typeof(clas);
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