package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import com.deep.framework.ast.lexer.TokenType;

import java.util.List;
import java.util.Objects;

public class Name extends Expression {
    private String identifier;
    private Name qualifier;
    private static Name name;

    public Name(String identifier) {
        super(null);
        this.identifier = identifier;
    }

    public static void parser(Node node) {
        CallableDeclaration.parser(node);
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n) -> {
            if (m instanceof Name a && Objects.nonNull(n) && n.equals(TokenType.DOT)) {
                name = a;
                node.replace(m, name);
            } else if (m.equals(TokenType.DOT) && Objects.nonNull(n)) {
                if (n instanceof Name) {
                    name.getChildrens().add(n);
                    node.getChildrens().remove(m);
                    node.getChildrens().remove(n);
                    list.remove(n);
                } else if (n instanceof CallableDeclaration){
                    name.getChildrens().add(n);
                    node.getChildrens().remove(m);
                    node.getChildrens().remove(n);
                    list.remove(n);
                }
            }
        });
    }

    public String toString() {
        return identifier;
    }
}
