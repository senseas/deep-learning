package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.DOT;

public class Name extends Expression {
    private String identifier;
    private Name qualifier;
    private TypeParametersExpression typeParameters;
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
            if (m instanceof Name a && Objects.nonNull(n) && n.equals(DOT)) {
                name = a;
                node.remove(a);
            } else if (m.equals(DOT) && Objects.nonNull(n)) {
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
            } else if (m instanceof Name a && Objects.nonNull(n) && n instanceof TypeParametersExpression b) {
                a.setTypeParameters(b);
                node.remove(b);
            }
        });
        CallableDeclaration.parser(node);
    }

    public void setTypeParameters(TypeParametersExpression typeParameters) {
        this.typeParameters = typeParameters;
    }

    public String toString() {
        return identifier;
    }
}