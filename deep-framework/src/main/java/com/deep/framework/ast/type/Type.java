package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import lombok.Data;

import java.util.Objects;

@Data
public class Type extends Node {
    private String identifier;
    private Name name;

    public Type(Name name) {
        this.identifier = name.getIdentifier();
        this.name = name;
        getChildrens().add(name);
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