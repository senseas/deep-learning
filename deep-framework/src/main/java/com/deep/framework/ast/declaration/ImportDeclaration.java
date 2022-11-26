package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;

import static com.deep.framework.ast.lexer.TokenType.IMPORT;

public class ImportDeclaration extends Declaration {
    private Node name;

    public ImportDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce(((list, a, b) -> {
            if (a.equals(IMPORT)) {
                ImportDeclaration declare = new ImportDeclaration(node);
                declare.setName(b);
                declare.setChildrens(node.getChildrens());
                declare.getChildrens().remove(a);
                node.getPrarent().replace(node, declare);
                list.clear();
            }
        }));
    }

    public void setName(Node name) {
        this.name = name;
    }
}