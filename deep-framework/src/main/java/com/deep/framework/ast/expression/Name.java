package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.DOT;
import static com.deep.framework.ast.lexer.TokenType.MUL;

@Data
public class Name extends Expression {
    private String identifier;
    private Name qualifier;
    private TypeParametersExpression typeParameters;

    private static Expression name;

    public Name() {
        super(null);
    }

    public Name(String identifier) {
        super(null);
        this.identifier = identifier;
    }

    public static void parser(Node node) {
        CallableDeclaration.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (Objects.nonNull(n)) {
                if (m instanceof Name a && n instanceof TypeParametersExpression b) {
                    a.setTypeParameters(b);
                    node.getChildrens().remove(b);
                    list.clear();
                } else if (m instanceof Name a && n.equals(DOT)) {
                    name = a;
                    node.getChildrens().remove(a);
                } else if (m.equals(DOT) && n instanceof Name) {
                    n.setPrarent(node);
                    n.getChildrens().add(name);
                    name.setPrarent(n);
                    node.getChildrens().removeAll(List.of(m, name));
                    name = (Expression) n;
                    list.remove(n);
                } else if (m.equals(DOT) && n instanceof CallableDeclaration c) {
                    n.setPrarent(node);
                    n.getChildrens().add(name);
                    name.setPrarent(n);
                    c.getName().getChildrens().add(name);
                    node.getChildrens().removeAll(List.of(m, name));
                    name = (Expression) n;
                    list.remove(n);
                } else if (m.equals(DOT) && n.equals(MUL)) {
                    n.setPrarent(node);
                    n.getChildrens().add(name);
                    name.setPrarent(n);
                    node.getChildrens().removeAll(List.of(m, name));
                    name = null;
                    list.remove(n);
                }
            }
        });
    }

    public String toString() {
        return identifier;
    }
}