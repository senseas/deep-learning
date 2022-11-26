package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.PACKAGE;

public class PackageDeclaration extends Declaration {
    private Name name;

    public PackageDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce(((list, a, b) -> {
            if (a.equals(PACKAGE)) {
                PackageDeclaration declare = new PackageDeclaration(node);
                declare.setName((Name) b);
                declare.setChildrens(node.getChildrens());
                declare.getChildrens().remove(a);
                node.getPrarent().replace(node, declare);
                list.clear();
            }
        }));
    }

    public void setName(Name name) {
        this.name = name;
    }
}