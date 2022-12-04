package com.deep.framework.ast.type;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.TokenType;
import lombok.Data;

import java.util.Objects;

@Data
public class Type extends Expression {
    private final String identifier;
    private Name name;

    public Type(Name name) {
        super(null);
        this.identifier = name.getIdentifier();
        this.name = name;
    }

    public Type(TokenType type) {
        super(null);
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