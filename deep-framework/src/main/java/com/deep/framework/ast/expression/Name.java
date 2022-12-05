package com.deep.framework.ast.expression;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.declaration.CallableDeclaration;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.type.ArrayType;
import lombok.Data;

import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.DOT;

@Data
public class Name extends Expression {
    private String identifier;
    private Name qualifier;
    private TypeParametersExpression typeParameters;

    private static Node name;

    public Name() {
        super(null);
    }

    public Name(String identifier) {
        super(null);
        this.identifier = identifier;
    }

    public Name(TokenType type) {
        super(null);
        setTokenType(type);
        this.identifier = type.getName();
    }

    public static void parser(Node node) {
        name = null;
        CallableDeclaration.parser(node);
        ArrayType.parser(node);
        Stream.of(node.getChildrens()).reduce((list, m, n) -> {
            if (Objects.nonNull(n)) {
                if (m instanceof Name a && n instanceof TypeParametersExpression b) {
                    a.setTypeParameters(b);
                    node.getChildrens().remove(b);
                    list.clear();
                } else if (m instanceof Name a && n.equals(DOT)) {
                    name = a;
                    node.getChildrens().remove(a);
                } else if (m.equals(DOT) && n instanceof CallableDeclaration c) {
                    c.setPrarent(node);
                    c.getName().getChildrens().add(name);
                    name.setPrarent(c.getName());
                    node.getChildrens().removeAll(m, name);
                    name = n;
                    list.remove(n);
                } else if (m.equals(DOT) && n instanceof ArrayAccessExpression c) {
                    c.setPrarent(node);
                    c.getExpression().getChildrens().add(name);
                    name.setPrarent(c.getExpression());
                    node.getChildrens().removeAll(m, name);
                    name = n;
                    list.remove(n);
                } else if (m.equals(DOT)) {
                    n.setPrarent(node);
                    n.getChildrens().add(name);
                    name.setPrarent(n);
                    node.getChildrens().removeAll(m, name);
                    name = n;
                    list.remove(n);
                }
            }
        });
    }

    public String toString() {
        return identifier;
    }
}