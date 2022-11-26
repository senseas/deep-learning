package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;

import java.util.Objects;

public class Type extends Node {
    private final String identifier;

    public Type(Name name) {
        this.identifier = name.getIdentifier();
    }

    public Type(TokenType type) {
        this.identifier = type.getName();
    }

    public static Type getType(Node node) {
        if (Objects.nonNull(node.getTokenType())) {
            return new PrimitiveType(node.getTokenType());
        } else {
            return new ReferenceType((Name) node);
        }
    }

    @Override
    public String toString() {
        return identifier;
    }
}