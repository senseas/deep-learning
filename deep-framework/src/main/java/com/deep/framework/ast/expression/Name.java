package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.DOT;

@Data
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
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (Objects.nonNull(n)) {
                if (m instanceof Name a && n instanceof TypeParametersExpression b) {
                    a.setTypeParameters(b);
                    node.remove(b);
                    list.clear();
                } else if (m instanceof Name a && n.equals(DOT)) {
                    name = a;
                    node.remove(a);
                } else if (m.equals(DOT) && n instanceof Name) {
                    n.setPrarent(node);
                    n.getChildrens().add(name);
                    name.setPrarent(n);
                    node.getChildrens().removeAll(List.of(m, name));
                    name = (Name) n;
                    list.remove(n);
                }
            }
        });
        CallableDeclaration.parser(node);
    }

    public String toString() {
        return identifier;
    }
}