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

    public Name() {
        super(null);
    }

    public Name(String identifier) {
        super(null);
        this.identifier = identifier;
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce((List list, Object m, Object n) -> {
            if (m instanceof Name a && Objects.nonNull(n) && n.equals(TokenType.DOT)) {
                name = a;
                node.remove(a);
            } else if (m.equals(TokenType.DOT) && Objects.nonNull(n)) {
                if (n instanceof Name b) {
                    b.setPrarent(node);
                    b.getChildrens().add(name);
                    name.setPrarent(b);
                    node.remove(m);
                    node.remove(name);
                    name = b;
                    list.remove(n);
                } else {
                    node.remove(m);
                    name.setPrarent(node);
                }
            }
        });
        CallableDeclaration.parser(node);
    }

    public String toString() {
        return identifier;
    }
}
