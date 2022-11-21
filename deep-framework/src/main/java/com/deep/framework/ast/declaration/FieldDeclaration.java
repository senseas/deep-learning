package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.expression.Name;
import com.deep.framework.ast.lexer.Token;
import com.deep.framework.ast.lexer.TokenType;
import com.deep.framework.ast.statement.Statement;
import com.deep.framework.ast.type.Type;
import lombok.Data;

import java.util.List;
import java.util.stream.Collectors;

@Data
public class FieldDeclaration extends Declaration {
    private List<TokenType> modifiers;
    private Type type;
    private Name name;

    public FieldDeclaration(Node prarent) {
        super(prarent);
    }

    public static void parser(Node node) {
        if (node instanceof MethodDeclaration) return;
        List.copyOf(node.getChildrens()).stream().map(a -> (Node) a).forEach((a) -> {
            if (a instanceof Statement) {
                FieldDeclaration fieldDeclare = new FieldDeclaration(node.getPrarent());
                List<TokenType> modifiers = a.getChildrens().stream().filter(e -> Field_Modifiers.contains(e)).map(o -> ((Token) o).getTokenType()).collect(Collectors.toList());
                a.getChildrens().removeAll(modifiers);

                fieldDeclare.setModifiers(modifiers);
                fieldDeclare.setType(new Type((Name) a.getChildrens().stream().findFirst().get()));
                fieldDeclare.setChildrens(a.getChildrens());
                node.replace(a, fieldDeclare);
            }
        });
    }
}
