package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.IMPORT;

public class ImportDeclaration extends Declaration {

    public ImportDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        Stream.of(node.getChildrens()).reduce(((list, a, b) -> {
            if (a.equals(IMPORT)) {
                ImportDeclaration importDeclaration = new ImportDeclaration(node);
                importDeclaration.setName((Name) b);
                importDeclaration.setChildrens(node.getChildrens());
                importDeclaration.remove(a);
                node.getPrarent().replace(node, importDeclaration);
                list.clear();
            }
        }));
    }
}