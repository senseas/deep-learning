package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.IMPORT;

public class ImportDeclaration extends Declaration {
    private Name name;

    public ImportDeclaration(Node prarent, Name name) {
        super(prarent);
        this.name = name;
        this.name.setPrarent(this);
        getChildrens().add(name);
    }

    public Node getName() {
        return name;
    }

    public static void parser(Node node) {
        if (node instanceof ImportDeclaration) return;
        Stream.of(node.getChildrens()).reduce2(((list, a, b) -> {
            if (a.equals(IMPORT)) {
                ImportDeclaration declare = new ImportDeclaration(node, (Name) b);
                node.getPrarent().replace(node, declare);
                list.clear();
            }
        }));
    }

}