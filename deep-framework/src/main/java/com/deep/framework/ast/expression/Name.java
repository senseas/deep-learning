package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;

public class Name extends Node {
    private String identifier;
    private Name qualifier;

    public Name(String identifier) {
        this.identifier = identifier;
    }

    @Override
    public String toString() {
        return identifier;
    }
}
