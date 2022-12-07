package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.PACKAGE;

public class PackageDeclaration extends Declaration {
    private Name name;

    public PackageDeclaration(Node prarent, Name name) {
        super(prarent);
        this.name = name;
        this.name.setPrarent(this);
        getChildrens().add(name);
    }

    public Name getName() {
        return name;
    }

    public static void parser(Node node) {
        if (node instanceof PackageDeclaration) return;
        Stream.of(node.getChildrens()).reduce(((list, a, b) -> {
            if (a.equals(PACKAGE)) {
                PackageDeclaration declare = new PackageDeclaration(node, (Name) b);
                node.getPrarent().replace(node, declare);
                list.clear();
            }
        }));
    }

}